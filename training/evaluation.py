# training/evaluation.py
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import wandb


def plot_metrics(metrics_history, config, file_base=None):
    """
    Plot metrics over training epochs.
    
    Args:
        metrics_history: List of metrics dictionaries
        config: Configuration object
        file_base: Optional file base name for saving
    
    Returns:
        Dictionary with metadata about the plots
    """
    # Create model identifier for saving
    from datetime import datetime
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifier = f"{config.model.model_pair}_{formatted_time}"
    
    # Ensure directories exist
    os.makedirs(config.logging.eval_dir, exist_ok=True)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_eval_dir = os.path.join(config.logging.drive_path, "evaluation")
        os.makedirs(drive_eval_dir, exist_ok=True)
    
    epochs_list = [m['epoch'] for m in metrics_history]
    
    # Plot accuracy and F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
    plt.plot(epochs_list, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    plt.plot(epochs_list, [m['auc_roc'] for m in metrics_history], '^-', label='AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Classification Metrics - {config.model.model_pair}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    class_metrics_path = os.path.join(config.logging.eval_dir, f'classification_metrics_{model_identifier}.png')
    plt.savefig(class_metrics_path)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        plt.savefig(os.path.join(drive_eval_dir, f'classification_metrics_{model_identifier}.png'))
    plt.close()
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, [m['pearson_correlation'] for m in metrics_history], 'o-', label='Pearson')
    plt.plot(epochs_list, [m['spearman_correlation'] for m in metrics_history], 's-', label='Spearman')
    plt.plot(epochs_list, [m['kendall_tau'] for m in metrics_history], '^-', label='Kendall Tau')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title(f'Correlation with True Reward - {config.model.model_pair}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    corr_metrics_path = os.path.join(config.logging.eval_dir, f'correlation_metrics_{model_identifier}.png')
    plt.savefig(corr_metrics_path)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        plt.savefig(os.path.join(drive_eval_dir, f'correlation_metrics_{model_identifier}.png'))
    plt.close()
    
    # Plot average rewards
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
    plt.plot(epochs_list, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
    plt.plot(epochs_list, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title(f'Average Predicted Rewards - {config.model.model_pair}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    rewards_path = os.path.join(config.logging.eval_dir, f'rewards_{model_identifier}.png')
    plt.savefig(rewards_path)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        plt.savefig(os.path.join(drive_eval_dir, f'rewards_{model_identifier}.png'))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, [m['loss'] for m in metrics_history], 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {config.model.model_pair}')
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(config.logging.eval_dir, f'loss_{model_identifier}.png')
    plt.savefig(loss_path)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        plt.savefig(os.path.join(drive_eval_dir, f'loss_{model_identifier}.png'))
    plt.close()
    
    # Display combined plot (for notebooks)
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs_list, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
    plt.plot(epochs_list, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    plt.plot(epochs_list, [m['auc_roc'] for m in metrics_history], '^-', label='AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Classification Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_list, [m['pearson_correlation'] for m in metrics_history], 'o-', label='Pearson')
    plt.plot(epochs_list, [m['spearman_correlation'] for m in metrics_history], 's-', label='Spearman')
    plt.plot(epochs_list, [m['kendall_tau'] for m in metrics_history], '^-', label='Kendall Tau')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Correlation with True Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
    plt.plot(epochs_list, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
    plt.plot(epochs_list, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Average Predicted Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_list, [m['loss'] for m in metrics_history], 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the combined figure
    combined_path = os.path.join(config.logging.eval_dir, f'combined_metrics_{model_identifier}.png')
    plt.savefig(combined_path, dpi=300)
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        plt.savefig(os.path.join(drive_eval_dir, f'combined_metrics_{model_identifier}.png'), dpi=300)
    
    try:
        if 'google.colab' in str(get_ipython()):
            plt.show()
        else:
            plt.close()
    except:
        plt.close()
        
    # Log to wandb if configured
    if config.logging.use_wandb and wandb.run is not None:
        wandb.log({
            "classification_metrics_plot": wandb.Image(class_metrics_path),
            "correlation_metrics_plot": wandb.Image(corr_metrics_path),
            "rewards_plot": wandb.Image(rewards_path),
            "loss_plot": wandb.Image(loss_path),
            "combined_metrics_plot": wandb.Image(combined_path)
        })
    
    # Get additional system info
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # Create comprehensive metadata
    metadata = {
        # Model and data information
        "model_pair": config.model.model_pair,
        "model_identifier": model_identifier,
        "dataset_file_base": file_base,
        
        # Dataset parameters
        "total_samples": config.dataset.num_samples,
        "train_samples": int(config.dataset.num_samples * config.training.train_test_split),
        "test_samples": int(config.dataset.num_samples * (1 - config.training.train_test_split)),
        
        # Training parameters
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.epochs,
        "max_new_tokens": config.dataset.max_new_tokens,
        "eval_interval": config.training.eval_interval,
        "seed": config.training.seed,
        
        # Generation parameters
        "temperature": config.dataset.temperature,
        "top_p": config.dataset.top_p,
        
        # System information
        "formatted_time": formatted_time,
        "gpu_info": gpu_info,
        
        # Final metrics
        "final_metrics": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                          for k, v in metrics_history[-1].items() if k != 'epoch'}
    }
    
    # Save metadata
    metadata_path = os.path.join(config.logging.eval_dir, f'metadata_{model_identifier}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_metadata_path = os.path.join(drive_eval_dir, f'metadata_{model_identifier}.json')
        with open(drive_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Log metadata to wandb
    if config.logging.use_wandb and wandb.run is not None:
        wandb.config.update(metadata)
    
    print(f"Plots and metadata saved to {config.logging.eval_dir}")
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        print(f"Plots and metadata also saved to {drive_eval_dir}")
    print(f"Unique identifier: {model_identifier}")
    
    return metadata


def analyze_score_distribution(reward_model, tokenizer, test_data, config):
    """
    Plot distribution of scores for toxic vs non-toxic content.
    
    Args:
        reward_model: Trained reward model
        tokenizer: Tokenizer for the model
        test_data: Test data dictionary with 'original' and 'detoxified' keys
        config: Configuration object
        
    Returns:
        Tuple of original and detoxified scores
    """
    device = next(reward_model.parameters()).device
    
    # Function to get scores
    def get_scores(texts, batch_size=8):
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.training.max_length
            ).to(device)
            with torch.no_grad():
                outputs = reward_model(**inputs)
            scores_batch = outputs.squeeze().cpu().tolist()
            
            # Handle single item case
            if not isinstance(scores_batch, list):
                scores_batch = [scores_batch]
                
            scores.extend(scores_batch)
        return scores
    
    # Get scores
    original_texts = [item['output'] for item in test_data['original']]
    detoxified_texts = [item['output'] for item in test_data['detoxified']]
    
    original_scores = get_scores(original_texts)
    detoxified_scores = get_scores(detoxified_texts)
    
    # Create model identifier for saving
    from datetime import datetime
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifier = f"{config.model.model_pair}_{formatted_time}"
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(original_scores, alpha=0.5, bins=20, label='Original (Toxic)', color='red')
    plt.hist(detoxified_scores, alpha=0.5, bins=20, label='Detoxified', color='green')
    plt.axvline(np.mean(original_scores), color='red', linestyle='--')
    plt.axvline(np.mean(detoxified_scores), color='green', linestyle='--')
    plt.xlabel('Reward Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Reward Scores - {config.model.model_pair}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to disk
    dist_path = os.path.join(config.logging.eval_dir, f'score_distribution_{model_identifier}.png')
    plt.savefig(dist_path)
    
    # Save to Google Drive if configured
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_eval_dir = os.path.join(config.logging.drive_path, "evaluation")
        plt.savefig(os.path.join(drive_eval_dir, f'score_distribution_{model_identifier}.png'))
    
    # Log to wandb if configured
    if config.logging.use_wandb and wandb.run is not None:
        wandb.log({
            "score_distribution": wandb.Image(dist_path),
            "original_scores": wandb.Histogram(original_scores),
            "detoxified_scores": wandb.Histogram(detoxified_scores),
            "mean_original_score": np.mean(original_scores),
            "mean_detoxified_score": np.mean(detoxified_scores),
            "score_difference": np.mean(detoxified_scores) - np.mean(original_scores)
        })
    
    try:
        if 'google.colab' in str(get_ipython()):
            plt.show()
        else:
            plt.close()
    except:
        plt.close()
    
    return original_scores, detoxified_scores


def analyze_errors(reward_model, tokenizer, test_data, config):
    """
    Analyze misclassified examples.
    
    Args:
        reward_model: Trained reward model
        tokenizer: Tokenizer for the model
        test_data: Test data dictionary with 'original' and 'detoxified' keys
        config: Configuration object
        
    Returns:
        Tuple of misclassified original and detoxified examples
    """
    device = next(reward_model.parameters()).device
    
    # Get scores for all examples
    original_texts = [item['output'] for item in test_data['original']]
    detoxified_texts = [item['output'] for item in test_data['detoxified']]
    
    # Function to get scores
    def get_scores(texts, batch_size=8):
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.training.max_length
            ).to(device)
            with torch.no_grad():
                outputs = reward_model(**inputs)
            batch_scores = outputs.squeeze().cpu().tolist()
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
        return scores
    
    original_scores = get_scores(original_texts)
    detoxified_scores = get_scores(detoxified_texts)
    
    # Determine threshold
    all_scores = original_scores + detoxified_scores
    threshold = np.mean(all_scores)
    
    # Find misclassified examples
    misclassified_original = [(i, score) for i, score in enumerate(original_scores) if score > threshold]
    misclassified_detoxified = [(i, score) for i, score in enumerate(detoxified_scores) if score <= threshold]
    
    # Create model identifier for saving
    from datetime import datetime
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifier = f"{config.model.model_pair}_{formatted_time}"
    
    # Print results
    print(f"Misclassified original examples: {len(misclassified_original)}/{len(original_scores)}")
    print(f"Misclassified detoxified examples: {len(misclassified_detoxified)}/{len(detoxified_scores)}")
    
    # Save to file
    error_file = os.path.join(config.logging.eval_dir, f'error_analysis_{model_identifier}.txt')
    
    with open(error_file, 'w') as f:
        f.write(f"Misclassified original examples: {len(misclassified_original)}/{len(original_scores)}\n")
        f.write(f"Misclassified detoxified examples: {len(misclassified_detoxified)}/{len(detoxified_scores)}\n\n")
        
        f.write("Misclassified original examples (classified as non-toxic):\n")
        for i, score in sorted(misclassified_original, key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Text: {test_data['original'][i]['output']}\n")
            f.write("-" * 80 + "\n")
        
        f.write("\nMisclassified detoxified examples (classified as toxic):\n")
        for i, score in sorted(misclassified_detoxified, key=lambda x: x[1])[:10]:
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Text: {test_data['detoxified'][i]['output']}\n")
            f.write("-" * 80 + "\n")
    
    # Save to Google Drive if configured
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_eval_dir = os.path.join(config.logging.drive_path, "evaluation")
        drive_error_file = os.path.join(drive_eval_dir, f'error_analysis_{model_identifier}.txt')
        
        with open(drive_error_file, 'w') as f:
            f.write(f"Misclassified original examples: {len(misclassified_original)}/{len(original_scores)}\n")
            f.write(f"Misclassified detoxified examples: {len(misclassified_detoxified)}/{len(detoxified_scores)}\n\n")
            
            f.write("Misclassified original examples (classified as non-toxic):\n")
            for i, score in sorted(misclassified_original, key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"Score: {score:.4f}\n")
                f.write(f"Text: {test_data['original'][i]['output']}\n")
                f.write("-" * 80 + "\n")
            
            f.write("\nMisclassified detoxified examples (classified as toxic):\n")
            for i, score in sorted(misclassified_detoxified, key=lambda x: x[1])[:10]:
                f.write(f"Score: {score:.4f}\n")
                f.write(f"Text: {test_data['detoxified'][i]['output']}\n")
                f.write("-" * 80 + "\n")
    
    # Log to wandb if configured
    if config.logging.use_wandb and wandb.run is not None:
        # Create tables for misclassified examples
        misclassified_orig_table = wandb.Table(columns=["Index", "Score", "Text"])
        for i, score in sorted(misclassified_original, key=lambda x: x[1], reverse=True)[:10]:
            misclassified_orig_table.add_data(i, score, test_data['original'][i]['output'])
        
        misclassified_detox_table = wandb.Table(columns=["Index", "Score", "Text"])
        for i, score in sorted(misclassified_detoxified, key=lambda x: x[1])[:10]:
            misclassified_detox_table.add_data(i, score, test_data['detoxified'][i]['output'])
        
        wandb.log({
            "misclassified_original": misclassified_orig_table,
            "misclassified_detoxified": misclassified_detox_table,
            "misclassification_rate_original": len(misclassified_original) / len(original_scores),
            "misclassification_rate_detoxified": len(misclassified_detoxified) / len(detoxified_scores),
            "error_analysis": wandb.Artifact(f"error_analysis_{model_identifier}", type="text")
        })
    
    print(f"Error analysis saved to {error_file}")
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        print(f"Error analysis also saved to {drive_error_file}")
    
    return misclassified_original, misclassified_detoxified