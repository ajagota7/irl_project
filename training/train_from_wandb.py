import sys
sys.path.append('/content/irl_project')

import os
import torch
import wandb
import json
import pandas as pd
from tqdm import tqdm

from config.config import Config
from data.dataset import DatasetGenerator
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics

# Configure the training
def train_with_wandb_dataset():
    # Set up basic configuration
    config = Config()
    config.model.model_pair = "gpt-neo-125M"
    config.training.epochs = 30
    config.training.batch_size = 16
    config.training.learning_rate = 5e-6
    config.training.eval_interval = 5
    config.logging.use_wandb = True
    config.logging.project_name = "irl-full-run"
    config.logging.experiment_name = "full-run-2000samples-from-wandb"
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
    
    print("Fetching datasets from WandB artifacts...")
    
    # Fetch the original dataset artifact
    artifact_name = "original_dataset_gpt-neo-125M_n2000_t15_temp1.0:latest"
    original_artifact = wandb_run.use_artifact(artifact_name, type='dataset')
    original_dir = original_artifact.download()
    original_file = os.path.join(original_dir, "gpt-neo-125M_n2000_t15_temp1.0_original.json")
    
    # Fetch the detoxified dataset artifact
    artifact_name = "detoxified_dataset_gpt-neo-125M_n2000_t15_temp1.0:latest"
    detoxified_artifact = wandb_run.use_artifact(artifact_name, type='dataset')
    detoxified_dir = detoxified_artifact.download()
    detoxified_file = os.path.join(detoxified_dir, "gpt-neo-125M_n2000_t15_temp1.0_detoxified.json")
    
    # Load datasets from downloaded files
    print("Loading datasets from WandB artifacts...")
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    with open(detoxified_file, 'r') as f:
        detoxified_data = json.load(f)
    
    print(f"Loaded {len(original_data)} samples from original dataset")
    print(f"Loaded {len(detoxified_data)} samples from detoxified dataset")
    
    # Set file_base for saving results
    file_base = "gpt-neo-125M_n2000_t15_temp1.0"
    
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
        "total_epochs": len(metrics_history)
    })
    
    # Finish wandb run
    wandb.finish()
    
    print(f"Training complete. Model saved to {config.logging.model_dir}")
    
    if config.logging.save_to_drive and os.path.exists('/content/drive'):
        drive_path = os.path.join(config.logging.drive_path, "models")
        print(f"Model also saved to Google Drive at {drive_path}")

if __name__ == "__main__":
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
    
    # Run training with WandB datasets
    train_with_wandb_dataset()