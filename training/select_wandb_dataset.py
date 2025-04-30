import sys

import os
import torch
import wandb
import json
import pandas as pd
from tqdm import tqdm
import argparse

from config.config import Config
from data.dataset import DatasetGenerator
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics

def get_available_artifacts(project_name, artifact_type="dataset"):
    """List all available artifacts of a given type in a project."""
    api = wandb.Api()
    artifacts = []
    
    # Get all artifacts of specified type
    for artifact in api.artifacts(type=artifact_type, project=project_name):
        artifacts.append({
            "name": artifact.name,
            "version": artifact.version,
            "full_name": f"{artifact.name}:{artifact.version}",
            "created_at": artifact.created_at,
            "size": artifact.size
        })
    
    return artifacts

def display_artifacts(artifacts):
    """Display artifacts in a readable format."""
    print("\nAvailable Artifacts:")
    print("-" * 80)
    print(f"{'#':3} | {'Name':<40} | {'Version':<8} | {'Created':<20} | {'Size (MB)':<10}")
    print("-" * 80)
    
    for i, artifact in enumerate(artifacts):
        print(f"{i:3} | {artifact['name'][:40]:<40} | {artifact['version']:<8} | "
              f"{artifact['created_at'][:20]:<20} | {artifact['size']/(1024*1024):<10.2f}")

def train_with_selected_datasets(original_artifact_name, detoxified_artifact_name, config=None):
    """Train a model using selected WandB datasets."""
    # Set up basic configuration if not provided
    if config is None:
        config = Config()
        config.model.model_pair = "gpt-neo-125M"
        config.training.epochs = 30
        config.training.batch_size = 16
        config.training.learning_rate = 5e-6
        config.training.eval_interval = 5
        config.logging.use_wandb = True
        config.logging.project_name = "irl-full-run"
        config.logging.experiment_name = f"custom-dataset-run-{original_artifact_name.split(':')[0]}"
        config.logging.save_to_drive = True
        config.logging.drive_path = "/content/drive/MyDrive/irl_project_neo"
    
    # Set up directories
    config.setup_directories()
    
    # Initialize wandb
    wandb_run = wandb.init(
        project=config.logging.project_name,
        name=config.logging.experiment_name,
        config=config.to_dict()
    )
    
    print(f"Fetching datasets from WandB artifacts...")
    print(f"Original dataset: {original_artifact_name}")
    print(f"Detoxified dataset: {detoxified_artifact_name}")
    
    # Extract base filename from artifact name
    file_base = original_artifact_name.split('_original')[0].split(':')[0]
    if "original_dataset_" in file_base:
        file_base = file_base.replace("original_dataset_", "")
    
    # Fetch the original dataset artifact
    original_artifact = wandb_run.use_artifact(original_artifact_name, type='dataset')
    original_dir = original_artifact.download()
    
    # Find the json file in the downloaded directory
    original_files = [f for f in os.listdir(original_dir) if f.endswith('.json')]
    if not original_files:
        raise ValueError(f"No JSON files found in downloaded original artifact: {original_dir}")
    original_file = os.path.join(original_dir, original_files[0])
    
    # Fetch the detoxified dataset artifact
    detoxified_artifact = wandb_run.use_artifact(detoxified_artifact_name, type='dataset')
    detoxified_dir = detoxified_artifact.download()
    
    # Find the json file in the downloaded directory
    detoxified_files = [f for f in os.listdir(detoxified_dir) if f.endswith('.json')]
    if not detoxified_files:
        raise ValueError(f"No JSON files found in downloaded detoxified artifact: {detoxified_dir}")
    detoxified_file = os.path.join(detoxified_dir, detoxified_files[0])
    
    # Load datasets from downloaded files
    print("Loading datasets from WandB artifacts...")
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    with open(detoxified_file, 'r') as f:
        detoxified_data = json.load(f)
    
    print(f"Loaded {len(original_data)} samples from original dataset")
    print(f"Loaded {len(detoxified_data)} samples from detoxified dataset")
    
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
    plot_metrics(metrics_history, config, file_base=file_base)
    
    # Log final metrics
    wandb.log({
        "training_complete": True,
        "final_metrics": metrics_history[-1],
        "total_epochs": len(metrics_history),
        "original_dataset": original_artifact_name,
        "detoxified_dataset": detoxified_artifact_name,
    })
    
    print(f"Training complete. Model saved to {config.logging.model_dir}")
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_path = os.path.join(config.logging.drive_path, "models")
        print(f"Model also saved to Google Drive at {drive_path}")
    
    return reward_model, metrics_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with datasets from WandB")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--project", type=str, default="irl-full-run", help="WandB project name")
    parser.add_argument("--original", type=str, help="Original dataset artifact (format: name:version)")
    parser.add_argument("--detoxified", type=str, help="Detoxified dataset artifact (format: name:version)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    
    # Print device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Only list artifacts if requested
    if args.list:
        artifacts = get_available_artifacts(args.project)
        display_artifacts(artifacts)
        print("\nTo train with a specific dataset, use:")
        print("python select_wandb_dataset.py --original artifact_name:version --detoxified artifact_name:version")
        sys.exit(0)
    
    # If no dataset specified, start interactive mode
    if not args.original or not args.detoxified:
        print("Interactive mode: Select datasets to use for training")
        artifacts = get_available_artifacts(args.project)
        display_artifacts(artifacts)
        
        try:
            print("\nSelect original dataset (enter number):")
            original_idx = int(input("> "))
            original_artifact = artifacts[original_idx]['full_name']
            
            print("\nSelect detoxified dataset (enter number):")
            detoxified_idx = int(input("> "))
            detoxified_artifact = artifacts[detoxified_idx]['full_name']
        except (ValueError, IndexError) as e:
            print(f"Error selecting datasets: {e}")
            sys.exit(1)
    else:
        original_artifact = args.original
        detoxified_artifact = args.detoxified
    
    # Set up configuration
    config = Config()
    config.model.model_pair = "gpt-neo-125M"
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.eval_interval = 5
    config.logging.use_wandb = True
    config.logging.project_name = args.project
    
    # Create experiment name from dataset names
    original_base = original_artifact.split(':')[0] 
    if 'original_dataset_' in original_base:
        original_base = original_base.replace('original_dataset_', '')
    config.logging.experiment_name = f"run-{original_base}"
    
    # Train with selected datasets
    train_with_selected_datasets(original_artifact, detoxified_artifact, config)