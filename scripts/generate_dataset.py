# scripts/generate_dataset.py
import argparse
import os
import torch
import wandb

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator


def parse_args():
    """Parse command line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate datasets for IRL")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset arguments
    parser.add_argument("--model_pair", type=str, default="gpt-neo-125M", 
                        help="Model pair to use (gpt-neo-125M, gpt-neo-2.7B, gpt-j-6B)")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=30, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0, 
                        help="Top-p for generation")
    parser.add_argument("--output_dir", type=str, default="./datasets", 
                        help="Output directory for datasets")
    
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


def generate_datasets():
    """Generate datasets for IRL."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update dataset cache directory if provided
    if args.output_dir:
        config.dataset.cache_dir = args.output_dir
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    import numpy as np
    np.random.seed(config.training.seed)
    
    # Set up directories
    config.setup_directories()
    
    # Initialize wandb
    if config.logging.use_wandb:
        run_name = f"generate_{config.model.model_pair}_{config.dataset.num_samples}samples"
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=config.to_dict(),
            mode=config.logging.wandb_mode
        )
    
    # Create dataset generator
    generator = DatasetGenerator(config)
    
    # Generate datasets
    print(f"Generating datasets for {config.model.model_pair}...")
    original_data, detoxified_data = generator.generate_datasets()
    
    # Analyze datasets
    print("Analyzing datasets...")
    analysis = generator.analyze_datasets()
    
    # Log to wandb if configured
    if config.logging.use_wandb and wandb.run is not None:
        wandb.log({
            "dataset_generation_complete": True,
            "dataset_analysis": analysis
        })
        
        # Log the datasets as artifacts
        dataset_artifact = wandb.Artifact(
            f"datasets_{config.model.model_pair}_{config.dataset.num_samples}samples", 
            type="dataset"
        )
        dataset_artifact.add_dir(config.dataset.cache_dir)
        wandb.log_artifact(dataset_artifact)
        
        # Finish wandb run
        wandb.finish()
    
    print(f"Dataset generation complete. Files saved to {config.dataset.cache_dir}")
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_path = os.path.join(config.logging.drive_path, "datasets")
        print(f"Datasets also saved to Google Drive at {drive_path}")


if __name__ == "__main__":
    generate_datasets()