# main.py
import os
import argparse
import wandb
import torch
import numpy as np
import json

from config.config import Config, get_config_from_args
from data.dataset import load_or_generate_dataset, DatasetGenerator
from models.reward_model import RewardModel
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics, analyze_score_distribution, analyze_errors
from utils.logging import setup_wandb, ExperimentLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Reward Model with IRL")
    
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
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0, 
                        help="Top-p for generation")
    parser.add_argument("--use_cached", action="store_true", 
                        help="Use cached datasets if available")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for dataset generation")
    parser.add_argument("--train_batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=5, 
                        help="Evaluation interval (epochs)")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="irl-experiments", 
                        help="Project name for wandb")
    parser.add_argument("--experiment_name", type=str, 
                        help="Experiment name for wandb")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use wandb for logging")
    parser.add_argument("--save_to_drive", action="store_true", 
                        help="Save results to Google Drive (for Colab)")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["generate", "train", "evaluate", "all"], 
                        help="Mode to run")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update config mode
    mode = args.mode
    
    # Set up directories
    config.setup_directories()
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Set up wandb
    if config.logging.use_wandb:
        setup_wandb(config)
    
    # Initialize experiment logger
    logger = ExperimentLogger(config)
    
    # Generate or load datasets
    if mode in ["generate", "all"]:
        print(f"Generating datasets for {config.model.model_pair}...")
        original_data, detoxified_data, file_base = load_or_generate_dataset(config)
        
        # Analyze datasets
        dataset_generator = DatasetGenerator(config)
        dataset_generator.original_data = original_data
        dataset_generator.detoxified_data = detoxified_data
        dataset_analysis = dataset_generator.analyze_datasets()
        
        # Log dataset analysis
        if config.logging.use_wandb:
            wandb.log({"dataset_analysis": dataset_analysis})
    
    # Skip training and evaluation if only generating datasets
    if mode == "generate":
        print("Dataset generation complete.")
        if config.logging.use_wandb:
            wandb.finish()
        return
    
    # Load datasets if not generated
    if mode in ["train", "evaluate"]:
        print(f"Loading datasets for {config.model.model_pair}...")
        dataset_generator = DatasetGenerator(config)
        original_data, detoxified_data = dataset_generator.load_datasets()
        file_base = dataset_generator.file_base
    
    # Initialize trainer
    trainer = RewardModelTrainer(config)
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
    
    # Train model
    if mode in ["train", "all"]:
        print(f"Training reward model for {config.model.model_pair}...")
        reward_model, metrics_history = trainer.train(train_data, test_data)
        
        # Log final metrics
        if config.logging.use_wandb:
            wandb.log({"final_metrics": metrics_history[-1]})
    
    # Skip evaluation if only training
    if mode == "train":
        print("Training complete.")
        if config.logging.use_wandb:
            wandb.finish()
        return
    
    # Load model if only evaluating
    if mode == "evaluate":
        print(f"Loading model for evaluation...")
        # TODO: Implement model loading
        reward_model = trainer.reward_model
        metrics_history = trainer.metrics_history
    
    # Evaluate model
    if mode in ["evaluate", "all"]:
        print(f"Evaluating model...")
        
        # Plot metrics
        plot_metrics(metrics_history, config, file_base=file_base)
        
        # Analyze score distribution
        original_scores, detoxified_scores = analyze_score_distribution(
            reward_model, trainer.tokenizer, test_data, config
        )
        
        # Analyze errors
        misclassified_original, misclassified_detoxified = analyze_errors(
            reward_model, trainer.tokenizer, test_data, config
        )
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Final accuracy: {metrics_history[-1]['accuracy']:.4f}")
        print(f"Final F1 score: {metrics_history[-1]['f1']:.4f}")
        print(f"Final AUC-ROC: {metrics_history[-1]['auc_roc']:.4f}")
        print(f"Final Pearson correlation: {metrics_history[-1]['pearson_correlation']:.4f}")
        print(f"Average reward difference: {metrics_history[-1]['reward_diff']:.4f}")
    
    # Finish logging
    if config.logging.use_wandb:
        wandb.finish()
    
    print("Done!")


if __name__ == "__main__":
    main()