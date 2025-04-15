# training/trainer.py
import os
import torch
import numpy as np
import json
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats

from models.reward_model import RewardModel


class RewardModelTrainer:
    """Trainer class for the reward model."""
    
    def __init__(self, config, model=None, tokenizer=None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            model: Optional pre-initialized model
            tokenizer: Optional pre-initialized tokenizer
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model if not provided
        if model is None or tokenizer is None:
            self._init_model_and_tokenizer()
        else:
            self.reward_model = model
            self.tokenizer = tokenizer
        
        # Initialize true reward model for evaluation
        self._init_true_reward_model()
        
        # Set random seed
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        
        # Create unique model identifier
        import time
        from datetime import datetime
        self.timestamp = int(time.time())
        self.formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_identifier = f"{config.model.model_pair}_{self.formatted_time}"
        
        # Set up model paths
        self.model_save_dir = os.path.join(config.logging.model_dir, self.model_identifier)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = []
    
    def _init_model_and_tokenizer(self):
        """Initialize the reward model and tokenizer."""
        original_model_path, _ = self.config.model.get_model_paths()
        
        print(f"Initializing reward model based on {original_model_path}...")
        self.reward_model = RewardModel(
            model_name=original_model_path,
            use_half_precision=self.config.model.use_half_precision,
            device=self.device,
            num_unfrozen_layers=1  # Unfreeze only the last layer
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
    
    def _init_true_reward_model(self):
        """Initialize the true reward model for evaluation."""
        print("Loading true reward model...")
        true_reward_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
        self.true_reward_tokenizer = RobertaTokenizer.from_pretrained(true_reward_model_name)
        self.true_reward_model = AutoModelForSequenceClassification.from_pretrained(
            true_reward_model_name,
            torch_dtype=torch.float16
        ).to(self.device)
    
    def prepare_data(self, original_data, detoxified_data):
        """
        Prepare data for training.
        
        Args:
            original_data: List of original (potentially toxic) samples
            detoxified_data: List of detoxified samples
            
        Returns:
            Train and test data splits
        """
        # Verify data
        assert len(original_data) == len(detoxified_data), "Original and detoxified data must have the same length"
        
        # Split data into train/test sets
        train_size = int(self.config.training.train_test_split * len(original_data))
        
        train_data = {
            'original': original_data[:train_size],
            'detoxified': detoxified_data[:train_size]
        }
        
        test_data = {
            'original': original_data[train_size:],
            'detoxified': detoxified_data[train_size:]
        }
        
        print(f"Training set: {len(train_data['original'])} samples")
        print(f"Test set: {len(test_data['original'])} samples")
        
        return train_data, test_data
    
    def data_loader(self, original_data, detoxified_data, batch_size):
        """
        Create batches of paired data.
        
        Args:
            original_data: Original (toxic) samples
            detoxified_data: Detoxified samples
            batch_size: Batch size
            
        Yields:
            Batches of paired data
        """
        assert len(original_data) == len(detoxified_data), "Both datasets should have the same length"
        
        indices = np.arange(len(original_data))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_original = [original_data[idx] for idx in batch_indices]
            batch_detoxified = [detoxified_data[idx] for idx in batch_indices]
            
            yield batch_original, batch_detoxified
    
    def max_margin_loss(self, original_rewards, detoxified_rewards, margin=None):
        """
        Compute max-margin loss.
        
        Args:
            original_rewards: Rewards for original (toxic) samples
            detoxified_rewards: Rewards for detoxified samples
            margin: Margin value (if None, use config value)
            
        Returns:
            Loss value
        """
        if margin is None:
            margin = self.config.training.margin
            
        # We want detoxified_rewards > original_rewards + margin
        reward_diff = detoxified_rewards - original_rewards
        loss = torch.clamp(margin - reward_diff, min=0)
        
        # Check for NaN and replace with zeros
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return loss.mean()
    
    def train(self, train_data, test_data):
        """
        Train the reward model.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Trained model and metrics history
        """
        # Initialize optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.reward_model.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
        
        # Training loop
        print("Starting training...")
        self.metrics_history = []
        
        # Log config to wandb
        if self.config.logging.use_wandb and wandb.run is not None:
            wandb.config.update(self.config.to_dict())
        
        for epoch in range(self.config.training.epochs):
            self.reward_model.train()
            epoch_losses = []
            
            # Progress bar for batches
            progress_bar = tqdm(
                self.data_loader(
                    train_data['original'], 
                    train_data['detoxified'], 
                    self.config.training.batch_size
                ),
                desc=f"Epoch {epoch+1}/{self.config.training.epochs}"
            )
            
            # Process batches
            for batch_original, batch_detoxified in progress_bar:
                optimizer.zero_grad()
                
                # Get original outputs
                original_texts = [item['output'] for item in batch_original]
                original_inputs = self.tokenizer(
                    original_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move everything to the correct device
                original_inputs = {k: v.to(self.device) for k, v in original_inputs.items()}
                
                original_rewards = self.reward_model(**original_inputs)
                
                # Get detoxified outputs
                detoxified_texts = [item['output'] for item in batch_detoxified]
                detoxified_inputs = self.tokenizer(
                    detoxified_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move everything to the correct device
                detoxified_inputs = {k: v.to(self.device) for k, v in detoxified_inputs.items()}
                
                detoxified_rewards = self.reward_model(**detoxified_inputs)
                
                # Compute loss
                loss = self.max_margin_loss(original_rewards, detoxified_rewards)
                
                # Check for NaN before backward
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or Inf detected in loss. Skipping batch.")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(), 
                    max_norm=self.config.training.grad_clip
                )
                
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Free up memory
                del original_inputs, detoxified_inputs
                torch.cuda.empty_cache()
            
            # Calculate average loss for epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            print(f"Epoch {epoch+1}/{self.config.training.epochs}, Loss: {avg_loss:.4f}")
            
            # Log loss to wandb
            if self.config.logging.use_wandb and wandb.run is not None:
                wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
            
            # Evaluate periodically
            if (epoch + 1) % self.config.training.eval_interval == 0 or epoch == self.config.training.epochs - 1:
                print(f"Evaluating at epoch {epoch+1}...")
                metrics = self.evaluate(test_data)
                
                # Add epoch and loss to metrics
                metrics['epoch'] = epoch + 1
                metrics['loss'] = avg_loss
                self.metrics_history.append(metrics)
                
                # Print metrics
                print(f"Metrics at epoch {epoch+1}:")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                
                # Log metrics to wandb
                if self.config.logging.use_wandb and wandb.run is not None:
                    wandb.log(metrics)
                
                # Save checkpoint if configured
                if self.config.logging.save_model:
                    self.save_checkpoint(epoch + 1)
        
        # Save the final model
        if self.config.logging.save_model:
            self.save_model()
        
        return self.reward_model, self.metrics_history
    
    def evaluate(self, test_data):
        """
        Evaluate the reward model.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        self.reward_model.eval()
        self.true_reward_model.eval()
        
        original_outputs = []
        detoxified_outputs = []
        ground_truth_labels = []  # 1 for original (toxic), 0 for detoxified (non-toxic)
        
        # Process in batches
        batch_size = self.config.training.batch_size
        
        with torch.no_grad():
            # Process original (toxic) examples in batches
            for i in range(0, len(test_data['original']), batch_size):
                batch = test_data['original'][i:i+batch_size]
                texts = [item['output'] for item in batch]
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get learned rewards
                rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                original_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([1] * len(batch))
            
            # Process detoxified examples in batches
            for i in range(0, len(test_data['detoxified']), batch_size):
                batch = test_data['detoxified'][i:i+batch_size]
                texts = [item['output'] for item in batch]
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get learned rewards
                rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                detoxified_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([0] * len(batch))
        
        # Compute true rewards using the ground truth model
        true_rewards = []
        all_texts = [test_data['original'][i]['output'] for i in range(len(test_data['original']))] + \
                    [test_data['detoxified'][i]['output'] for i in range(len(test_data['detoxified']))]
        
        # Process in batches
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            
            # Tokenize for the true reward model
            inputs = self.true_reward_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.training.max_length
            ).to(self.device)
            
            # Get true rewards
            logits = self.true_reward_model(**inputs).logits
            
            # Use the first logit (non-toxic) as the reward
            batch_rewards = logits[:, 0].cpu().tolist()
            true_rewards.extend(batch_rewards)
        
        # Get all outputs together
        all_outputs = original_outputs + detoxified_outputs
        
        # Compute metrics
        metrics = {}
        
        # Convert learned rewards to binary predictions
        # Higher reward should indicate less toxic (more detoxified)
        threshold = np.mean(all_outputs)  # Simple threshold
        learned_predictions = (np.array(all_outputs) > threshold).astype(int)
        learned_predictions = 1 - learned_predictions  # Invert to match ground truth (1=toxic)
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(ground_truth_labels, learned_predictions)
        
        # F1 Score
        metrics['f1'] = f1_score(ground_truth_labels, learned_predictions)
        
        # AUC-ROC
        metrics['auc_roc'] = roc_auc_score(ground_truth_labels, [-x for x in all_outputs])  # Invert for ROC
        
        # Correlation with true rewards
        metrics['pearson_correlation'] = np.corrcoef([-x for x in all_outputs], true_rewards)[0, 1]
        metrics['spearman_correlation'] = stats.spearmanr([-x for x in all_outputs], true_rewards).correlation
        metrics['kendall_tau'] = stats.kendalltau([-x for x in all_outputs], true_rewards).correlation
        
        # Average predicted rewards
        metrics['avg_original_reward'] = np.mean(original_outputs)
        metrics['avg_detoxified_reward'] = np.mean(detoxified_outputs)
        metrics['reward_diff'] = metrics['avg_detoxified_reward'] - metrics['avg_original_reward']
        
        # Return metrics
        return metrics
    
    def save_checkpoint(self, epoch):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.model_save_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        self.reward_model.save(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save metrics
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                 for k, v in self.metrics_history[-1].items()},
                f, indent=2
            )
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # This will upload the files to wandb
            checkpoint_artifact = wandb.Artifact(
                f"model-checkpoint-{epoch}", 
                type="model",
                metadata=self.metrics_history[-1]
            )
            checkpoint_artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(checkpoint_artifact)
        
        # Save to Google Drive if configured
        if self.config.logging.save_to_drive and os.path.exists('/content/drive'):
            try:
                drive_checkpoint_dir = os.path.join(
                    self.config.logging.drive_path, 
                    "models",
                    self.model_identifier,
                    f"checkpoint-{epoch}"
                )
                os.makedirs(drive_checkpoint_dir, exist_ok=True)
                
                # Copy model
                drive_model_path = os.path.join(drive_checkpoint_dir, "model.pt")
                self.reward_model.save(drive_model_path)
                
                # Copy tokenizer
                self.tokenizer.save_pretrained(drive_checkpoint_dir)
                
                # Copy metrics
                drive_metrics_path = os.path.join(drive_checkpoint_dir, "metrics.json")
                with open(drive_metrics_path, "w") as f:
                    json.dump(
                        {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in self.metrics_history[-1].items()},
                        f, indent=2
                    )
            except Exception as e:
                print(f"Could not save checkpoint to Google Drive: {e}")
    
    def save_model(self):
        """Save the final model."""
        # Create model directory
        model_path = os.path.join(self.model_save_dir, "model.pt")
        
        # Save model
        self.reward_model.save(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.model_save_dir)
        
        # Save all metrics
        metrics_path = os.path.join(self.model_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                  for k, v in m.items()} for m in self.metrics_history],
                f, indent=2
            )
        
        # Save config
        config_path = os.path.join(self.model_save_dir, "config.yaml")
        self.config.save(config_path)
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # This will upload the files to wandb
            model_artifact = wandb.Artifact(
                f"model-final", 
                type="model",
                metadata=self.metrics_history[-1] if self.metrics_history else {}
            )
            model_artifact.add_dir(self.model_save_dir)
            wandb.log_artifact(model_artifact)
        
        # Save to Google Drive if configured
        if self.config.logging.save_to_drive and os.path.exists('/content/drive'):
            try:
                drive_model_dir = os.path.join(
                    self.config.logging.drive_path, 
                    "models",
                    self.model_identifier
                )
                os.makedirs(drive_model_dir, exist_ok=True)
                
                # Copy model
                drive_model_path = os.path.join(drive_model_dir, "model.pt")
                self.reward_model.save(drive_model_path)
                
                # Copy tokenizer
                self.tokenizer.save_pretrained(drive_model_dir)
                
                # Copy metrics
                drive_metrics_path = os.path.join(drive_model_dir, "metrics.json")
                with open(drive_metrics_path, "w") as f:
                    json.dump(
                        [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                          for k, v in m.items()} for m in self.metrics_history],
                        f, indent=2
                    )
                
                # Copy config
                drive_config_path = os.path.join(drive_model_dir, "config.yaml")
                self.config.save(drive_config_path)
                
                print(f"Model and metrics also saved to Google Drive at {drive_model_dir}")
            except Exception as e:
                print(f"Could not save model to Google Drive: {e}")
        
        print(f"Training complete. Model saved to {self.model_save_dir}")