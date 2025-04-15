# scripts/evaluate_model.py
import argparse
import os
import torch
import wandb

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator
from models.reward_model import RewardModel
from training.trainer import RewardModelTrainer
from training.evaluation import (
    plot_metrics, analyze_score_distribution, analyze_errors
)
from utils.logging import setup_wandb
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate reward model")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model_pair", type=str, default="gpt-neo-125M", 
                        help="Model pair to use (gpt-neo-125M, gpt-neo-2.7B, gpt-j-6B)")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model file")
    parser.add_argument("--tokenizer_path", type=str, 
                        help="Path to tokenizer directory (if different from model path)")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default="./datasets", 
                        help="Directory with datasets")
    parser.add_argument("--dataset_file_base", type=str, 
                        help="Base filename for dataset files")
    
    # Evaluation arguments
    parser.add_argument("--eval_dir", type=str, default="./evaluation", 
                        help="Output directory for evaluation results")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="irl-experiments", 
                        help="Project name for wandb")
    parser.add_argument("--experiment_name", type=str, 
                        help="Experiment name for wandb")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use wandb for logging")
    parser.add_argument("--save_to_drive", action="store_true", 
                        help="Save results to Google Drive (for Colab)")
    parser.add_argument("--drive_path", type=str, 
                        default="/content/drive/MyDrive/irl_experiments", 
                        help="Path to Google Drive directory (for Colab)")
    
    return parser.parse_args()


def evaluate_model():
    """Evaluate a trained reward model."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update evaluation directory if provided
    if args.eval_dir:
        config.logging.eval_dir = args.eval_dir
    
    # Update dataset directory if provided
    if args.dataset_dir:
        config.dataset.cache_dir = args.dataset_dir
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    import numpy as np
    np.random.seed(config.training.seed)
    
    # Set up directories
    config.setup_directories()
    
    # Initialize wandb
    if config.logging.use_wandb:
        run_name = (args.experiment_name or 
                   f"evaluate_{config.model.model_pair}")
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=config.to_dict(),
            mode=config.logging.wandb_mode
        )
    
    # Load datasets
    print(f"Loading datasets from {config.dataset.cache_dir}...")
    generator = DatasetGenerator(config)
    
    # If dataset_file_base is provided, use it
    if args.dataset_file_base:
        generator.file_base = args.dataset_file_base
        
    # Load datasets
    original_data, detoxified_data = generator.load_datasets()
    
    # Load the tokenizer
    tokenizer_path = args.tokenizer_path or os.path.dirname(args.model_path)
    original_model_path, _ = config.model.get_model_paths()
    
    try:
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        print(f"Falling back to original model tokenizer from {original_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    reward_model = RewardModel.load(args.model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    reward_model.eval()
    
    # Create trainer for evaluation utilities
    trainer = RewardModelTrainer(config, model=reward_model, tokenizer=tokenizer)
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = trainer.evaluate(test_data)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Add metrics to history for plotting
    metrics_history = [metrics]
    
    # Analyze score distribution
    print("Analyzing score distribution...")
    original_scores, detoxified_scores = analyze_score_distribution(
        reward_model, tokenizer, test_data, config
    )
    
    # Analyze errors
    print("Analyzing misclassifications...")
    misclassified_original, misclassified_detoxified = analyze_errors(
        reward_model, tokenizer, test_data, config
    )
    
    # Log to wandb if configured
    if config.logging.use_wandb and wandb.run is not None:
        wandb.log({
            "evaluation_complete": True,
            "evaluation_metrics": metrics,
            "misclassified_original_count": len(misclassified_original),
            "misclassified_detoxified_count": len(misclassified_detoxified),
            "original_mean_score": np.mean(original_scores),
            "detoxified_mean_score": np.mean(detoxified_scores),
            "score_difference": np.mean(detoxified_scores) - np.mean(original_scores)
        })
        
        # Finish wandb run
        wandb.finish()
    
    print(f"Evaluation complete. Results saved to {config.logging.eval_dir}")
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_path = os.path.join(config.logging.drive_path, "evaluation")
        print(f"Results also saved to Google Drive at {drive_path}")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()