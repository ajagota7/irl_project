# scripts/train_model.py
import argparse
import os
import torch
import wandb

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics
from utils.logging import setup_wandb


def parse_args():
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train reward model with IRL")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset arguments
    parser.add_argument("--model_pair", type=str, default="gpt-neo-125M", 
                        help="Model pair to use (gpt-neo-125M, gpt-neo-2.7B, gpt-j-6B)")
    parser.add_argument("--dataset_dir", type=str, default="./datasets", 
                        help="Directory with datasets")
    parser.add_argument("--dataset_file_base", type=str, 
                        help="Base filename for dataset files")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=5, 
                        help="Evaluation interval (epochs)")
    parser.add_argument("--margin", type=float, default=0.1, 
                        help="Margin for max-margin loss")
    parser.add_argument("--unfrozen_layers", type=int, default=1, 
                        help="Number of unfrozen layers in the model")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./models", 
                        help="Output directory for model")
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


def train_model():
    """Train reward model with IRL."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update model directory if provided
    if args.output_dir:
        config.logging.model_dir = args.output_dir
    
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
                   f"train_{config.model.model_pair}_{config.training.epochs}epochs")
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
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = RewardModelTrainer(config)
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
    
    # Train model
    print(f"Training reward model for {config.model.model_pair}...")
    reward_model, metrics_history = trainer.train(train_data, test_data)
    
    # Plot metrics
    print("Generating training metrics plots...")
    plot_metrics(metrics_history, config, file_base=generator.file_base)
    
    # Log final metrics
    if config.logging.use_wandb and wandb.run is not None:
        wandb.log({
            "training_complete": True,
            "final_metrics": metrics_history[-1],
            "total_epochs": len(metrics_history)
        })
        
        # Finish wandb run
        wandb.finish()
    
    print(f"Training complete. Model saved to {config.logging.model_dir}")
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_path = os.path.join(config.logging.drive_path, "models")
        print(f"Model also saved to Google Drive at {drive_path}")
    
    return reward_model, trainer.tokenizer, metrics_history


if __name__ == "__main__":
    train_model()