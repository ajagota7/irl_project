import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from models.reward_model import RewardModel


class PairwiseDataset(Dataset):
    """Dataset for pairwise comparison training."""
    
    def __init__(self, tokenizer, chosen_data, rejected_data, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            chosen_data: List of preferred completions
            rejected_data: List of dispreferred completions
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.chosen_data = chosen_data
        self.rejected_data = rejected_data
        self.max_length = max_length
        
        assert len(chosen_data) == len(rejected_data), "Chosen and rejected data must have the same length"
    
    def __len__(self):
        return len(self.chosen_data)
    
    def __getitem__(self, idx):
        chosen_item = self.chosen_data[idx]
        rejected_item = self.rejected_data[idx]
        
        # Combine prompt and output
        chosen_text = chosen_item["prompt"] + chosen_item["output"]
        rejected_text = rejected_item["prompt"] + rejected_item["output"]
        
        # Tokenize
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        chosen_input_ids = chosen_encodings.input_ids.squeeze()
        chosen_attention_mask = chosen_encodings.attention_mask.squeeze()
        rejected_input_ids = rejected_encodings.input_ids.squeeze()
        rejected_attention_mask = rejected_encodings.attention_mask.squeeze()
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }


class FreshRewardModelTrainer:
    """Trainer for reward models."""
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, base_data, checkpoint_data):
        """
        Prepare data for training.
        
        Args:
            base_data: Data from the base model
            checkpoint_data: Data from the checkpoint model
            
        Returns:
            Dictionary with train and test data
        """
        # Shuffle data with the same seed for reproducibility
        np.random.seed(self.config.training.seed)
        indices = np.random.permutation(len(base_data))
        
        # Split into train and test
        split_idx = int(len(indices) * self.config.training.train_test_split)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Create train and test sets
        train_base = [base_data[i] for i in train_indices]
        train_checkpoint = [checkpoint_data[i] for i in train_indices]
        test_base = [base_data[i] for i in test_indices]
        test_checkpoint = [checkpoint_data[i] for i in test_indices]
        
        return {
            "train": {
                "base": train_base,
                "checkpoint": train_checkpoint
            },
            "test": {
                "base": test_base,
                "checkpoint": test_checkpoint
            }
        }
    
    def train(self, data):
        """
        Train the reward model.
        
        Args:
            data: Dictionary with train and test data
            
        Returns:
            Trained model and metrics history
        """
        # Initialize model
        model = RewardModel(
            self.config.model.base_model,
            use_half_precision=self.config.model.use_half_precision,
            device=self.device,
            num_unfrozen_layers=self.config.training.unfrozen_layers
        )
        
        # Create datasets
        train_dataset = PairwiseDataset(
            self.tokenizer,
            data["train"]["checkpoint"],  # Checkpoint data is preferred (less toxic)
            data["train"]["base"],        # Base data is dispreferred (more toxic)
            max_length=self.config.training.max_length
        )
        
        test_dataset = PairwiseDataset(
            self.tokenizer,
            data["test"]["checkpoint"],
            data["test"]["base"],
            max_length=self.config.training.max_length
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        # Initialize optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
        
        # Initialize scheduler
        total_steps = len(train_dataloader) * self.config.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # Initialize metrics history
        metrics_history = []
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}"):
                # Move batch to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Forward pass
                chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
                
                # Compute loss (higher reward for chosen, lower for rejected)
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update loss
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_dataloader)
            
            # Evaluation
            if (epoch + 1) % self.config.training.eval_interval == 0 or epoch == self.config.training.epochs - 1:
                metrics = self.evaluate(model, test_dataloader)
                metrics["epoch"] = epoch + 1
                metrics["train_loss"] = train_loss
                metrics_history.append(metrics)
                
                print(f"Epoch {epoch+1}/{self.config.training.epochs}, Train Loss: {train_loss:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}, Pearson: {metrics['pearson_correlation']:.4f}")
                print(f"  Score Difference: {metrics['score_difference']:.4f}")
        
        # Save model
        if self.config.logging.save_model:
            model_path = os.path.join(self.config.logging.model_dir, f"reward_model_{self.config.model.base_model.split('/')[-1]}.pt")
            model.save(model_path)
            print(f"Model saved to {model_path}")
        
        return model, metrics_history
    
    def evaluate(self, model, dataloader):
        """
        Evaluate the model.
        
        Args:
            model: Reward model
            dataloader: Test dataloader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Forward pass
                chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
                
                # Store rewards
                all_chosen_rewards.extend(chosen_rewards.cpu().numpy())
                all_rejected_rewards.extend(rejected_rewards.cpu().numpy())
                
                # Make predictions (1 if chosen > rejected, 0 otherwise)
                predictions = (chosen_rewards > rejected_rewards).int().cpu().numpy()
                all_predictions.extend(predictions)
                
                # Labels are always 1 (chosen should always be preferred)
                all_labels.extend([1] * len(predictions))
        
        # Convert to numpy arrays
        all_chosen_rewards = np.array(all_chosen_rewards)
        all_rejected_rewards = np.array(all_rejected_rewards)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        # Calculate AUC-ROC
        reward_diffs = all_chosen_rewards - all_rejected_rewards
        auc_roc = roc_auc_score(all_labels, reward_diffs)
        
        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(reward_diffs, all_labels)
        
        # Calculate average score difference
        score_difference = np.mean(all_chosen_rewards) - np.mean(all_rejected_rewards)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions, labels=[0, 1]).ravel()
        
        # Return metrics
        return {
            "accuracy": accuracy,
            "f1": f1,
            "auc_roc": auc_roc,
            "pearson_correlation": pearson_corr,
            "score_difference": score_difference,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "mean_chosen_reward": float(np.mean(all_chosen_rewards)),
            "mean_rejected_reward": float(np.mean(all_rejected_rewards)),
        }
    
    def analyze_scores(self, model, data):
        """
        Analyze scores from the model.
        
        Args:
            model: Reward model
            data: Dictionary with test data
            
        Returns:
            Dictionary with score analysis
        """
        model.eval()
        
        # Get test data
        test_base = data["test"]["base"]
        test_checkpoint = data["test"]["checkpoint"]
        
        # Calculate scores for base and checkpoint data
        base_scores = []
        checkpoint_scores = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = self.config.training.batch_size
            
            # Base data
            for i in range(0, len(test_base), batch_size):
                batch = test_base[i:i+batch_size]
                texts = [item["prompt"] + item["output"] for item in batch]
                
                encodings = self.tokenizer(
                    texts,
                    max_length=self.config.training.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                scores = model(encodings.input_ids, encodings.attention_mask)
                base_scores.extend(scores.cpu().numpy())
            
            # Checkpoint data
            for i in range(0, len(test_checkpoint), batch_size):
                batch = test_checkpoint[i:i+batch_size]
                texts = [item["prompt"] + item["output"] for item in batch]
                
                encodings = self.tokenizer(
                    texts,
                    max_length=self.config.training.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                scores = model(encodings.input_ids, encodings.attention_mask)
                checkpoint_scores.extend(scores.cpu().numpy())
        
        # Convert to numpy arrays
        base_scores = np.array(base_scores)
        checkpoint_scores = np.array(checkpoint_scores)
        
        # Calculate statistics
        base_mean = np.mean(base_scores)
        checkpoint_mean = np.mean(checkpoint_scores)
        score_difference = checkpoint_mean - base_mean
        
        # Find misclassified examples
        misclassified_base = []
        misclassified_checkpoint = []
        
        for i in range(len(test_base)):
            if base_scores[i] >= checkpoint_scores[i]:
                misclassified_base.append({
                    "index": i,
                    "prompt": test_base[i]["prompt"],
                    "output": test_base[i]["output"],
                    "score": float(base_scores[i]),
                    "checkpoint_score": float(checkpoint_scores[i])
                })
        
        for i in range(len(test_checkpoint)):
            if checkpoint_scores[i] <= base_scores[i]:
                misclassified_checkpoint.append({
                    "index": i,
                    "prompt": test_checkpoint[i]["prompt"],
                    "output": test_checkpoint[i]["output"],
                    "score": float(checkpoint_scores[i]),
                    "base_score": float(base_scores[i])
                })
        
        # Return analysis
        return {
            "base_scores": base_scores.tolist(),
            "checkpoint_scores": checkpoint_scores.tolist(),
            "base_mean": float(base_mean),
            "checkpoint_mean": float(checkpoint_mean),
            "score_difference": float(score_difference),
            "misclassified_base": misclassified_base,
            "misclassified_checkpoint": misclassified_checkpoint,
            "misclassification_rate_base": len(misclassified_base) / len(test_base),
            "misclassification_rate_checkpoint": len(misclassified_checkpoint) / len(test_checkpoint)
        }
    
    def plot_metrics(self, metrics_history, file_base=None):
        """
        Plot training metrics.
        
        Args:
            metrics_history: List of metrics dictionaries
            file_base: Base filename for saving plots
            
        Returns:
            Dictionary with plot paths
        """
        # Extract metrics
        epochs = [m["epoch"] for m in metrics_history]
        train_loss = [m["train_loss"] for m in metrics_history]
        accuracy = [m["accuracy"] for m in metrics_history]
        f1 = [m["f1"] for m in metrics_history]
        auc_roc = [m["auc_roc"] for m in metrics_history]
        pearson = [m["pearson_correlation"] for m in metrics_history]
        score_diff = [m["score_difference"] for m in metrics_history]
        
        # Create figure for metrics
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, accuracy, 'o-', label='Accuracy')
        plt.plot(epochs, f1, 's-', label='F1 Score')
        plt.plot(epochs, auc_roc, '^-', label='AUC-ROC')
        plt.plot(epochs, pearson, 'D-', label='Pearson Correlation')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        metrics_plot_path = os.path.join(self.config.logging.eval_dir, f"{file_base}_metrics.png" if file_base else "metrics.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        
        # Create figure for loss and score difference
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, 'o-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, score_diff, 's-', label='Score Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Score Difference')
        plt.title('Checkpoint - Base Score Difference')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        loss_plot_path = os.path.join(self.config.logging.eval_dir, f"{file_base}_loss.png" if file_base else "loss.png")
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        
        return {
            "metrics_plot": metrics_plot_path,
            "loss_plot": loss_plot_path
        }
    
    def plot_score_distribution(self, score_analysis, file_base=None):
        """
        Plot score distributions.
        
        Args:
            score_analysis: Dictionary with score analysis
            file_base: Base filename for saving plots
            
        Returns:
            Path to the plot
        """
        # Extract scores
        base_scores = score_analysis["base_scores"]
        checkpoint_scores = score_analysis["checkpoint_scores"]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        plt.hist(base_scores, bins=30, alpha=0.5, label='Base Model')
        plt.hist(checkpoint_scores, bins=30, alpha=0.5, label='Checkpoint Model')
        
        # Add vertical lines for means
        plt.axvline(score_analysis["base_mean"], color='blue', linestyle='dashed', linewidth=2, label=f'Base Mean: {score_analysis["base_mean"]:.4f}')
        plt.axvline(score_analysis["checkpoint_mean"], color='orange', linestyle='dashed', linewidth=2, label=f'Checkpoint Mean: {score_analysis["checkpoint_mean"]:.4f}')
        
        plt.xlabel('Reward Score')
        plt.ylabel('Frequency')
        plt.title(f'Score Distribution (Difference: {score_analysis["score_difference"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plot_path = os.path.join(self.config.logging.eval_dir, f"{file_base}_score_distribution.png" if file_base else "score_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return plot_path 