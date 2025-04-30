import os
import sys
import argparse
import torch
import wandb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.fresh_config import FreshConfig, get_config_from_args
from data.fresh_dataset import FreshDatasetGenerator
from training.fresh_trainer import FreshRewardModelTrainer


def parse_args():
    """Parse command line arguments for dataset evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate datasets with IRL reward model extraction")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset arguments
    parser.add_argument("--cache_dir", type=str, default="./datasets",
                        help="Directory with cached datasets")
    parser.add_argument("--dataset_results", type=str, default=None,
                        help="Path to dataset generation results JSON file")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--unfrozen_layers", type=int, default=1,
                        help="Number of unfrozen layers in the model")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="checkpoint-evaluation",
                        help="Project name for wandb")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for wandb")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--eval_dir", type=str, default="./evaluation",
                        help="Directory for evaluation results")
    
    return parser.parse_args()


def convert_for_json(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def evaluate_dataset(config, dataset_info):
    """
    Evaluate a single dataset.
    
    Args:
        config: Configuration object
        dataset_info: Dictionary with dataset information
        
    Returns:
        Dictionary with evaluation results
    """
    checkpoint_folder = dataset_info["checkpoint_folder"]
    checkpoint_num = dataset_info["checkpoint_num"]
    file_base = dataset_info["file_base"]
    
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_folder} (Epoch {checkpoint_num})")
    print(f"{'='*80}")
    
    try:
        # Create a unique experiment name for this checkpoint
        experiment_name = f"{config.model.checkpoint_repo.split('/')[-1]}_checkpoint-{checkpoint_num}"
        
        # Initialize wandb if configured
        if config.logging.use_wandb:
            wandb_run = wandb.init(
                project=config.logging.project_name,
                name=experiment_name,
                config=config.to_dict(),
                reinit=True
            )
        else:
            wandb_run = None
        
        # Initialize dataset generator
        dataset_generator = FreshDatasetGenerator(config)
        dataset_generator.file_base = file_base
        
        # Check if datasets exist
        if not dataset_generator.dataset_exists():
            raise FileNotFoundError(f"Datasets for {checkpoint_folder} not found")
        
        # Load datasets
        base_data, checkpoint_data = dataset_generator.load_datasets()
        
        # Analyze datasets
        print("Analyzing datasets...")
        dataset_analysis = dataset_generator.analyze_datasets()
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = FreshRewardModelTrainer(config)
        
        # Prepare data
        data = trainer.prepare_data(base_data, checkpoint_data)
        
        # Train model
        print(f"Training reward model for {checkpoint_folder}...")
        model, metrics_history = trainer.train(data)
        
        # Get final metrics
        final_metrics = metrics_history[-1] if metrics_history else {}
        
        # Fix numpy array formatting issues
        for key, value in final_metrics.items():
            if isinstance(value, np.ndarray):
                final_metrics[key] = float(value)
        
        # Calculate score difference
        score_analysis = trainer.analyze_scores(model, data)
        score_difference = score_analysis["score_difference"]
        
        # Calculate misclassification rates
        misclassification_rate_base = trainer.calculate_misclassification_rate(model, data, "base")
        misclassification_rate_checkpoint = trainer.calculate_misclassification_rate(model, data, "checkpoint")
        
        # Plot metrics
        print("Generating training metrics plots...")
        plot_paths = trainer.plot_metrics(metrics_history, file_base)
        
        # Plot score distribution
        print("Generating score distribution plot...")
        score_plot_path = trainer.plot_score_distribution(score_analysis, file_base)
        
        # Log to wandb if configured
        if wandb_run:
            # Log metrics
            wandb.log({
                "accuracy": final_metrics.get("accuracy", 0),
                "f1": final_metrics.get("f1", 0),
                "auc_roc": final_metrics.get("auc_roc", 0),
                "pearson_correlation": final_metrics.get("pearson_correlation", 0),
                "score_difference": score_difference,
                "misclassification_rate_base": misclassification_rate_base,
                "misclassification_rate_checkpoint": misclassification_rate_checkpoint
            })
            
            # Log plots
            wandb.log({
                "metrics_plot": wandb.Image(plot_paths["metrics_plot"]),
                "loss_plot": wandb.Image(plot_paths["loss_plot"]),
                "score_distribution": wandb.Image(score_plot_path)
            })
            
            # Finish wandb run
            wandb.finish()
        
        # Return results
        return {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "final_metrics": final_metrics,
            "score_difference": float(score_difference),
            "misclassification_rate_base": float(misclassification_rate_base),
            "misclassification_rate_checkpoint": float(misclassification_rate_checkpoint),
            "dataset_analysis": dataset_analysis,
            "plot_paths": plot_paths,
            "score_plot_path": score_plot_path
        }
        
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_folder}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "error": str(e)
        }


def main():
    """Main function for dataset evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = get_config_from_args(args)
    
    # Create evaluation directory
    os.makedirs(config.logging.eval_dir, exist_ok=True)
    
    # Load dataset generation results
    if args.dataset_results:
        with open(args.dataset_results, 'r') as f:
            dataset_results = json.load(f)
    else:
        # Find the most recent dataset generation results
        results_files = [f for f in os.listdir(config.dataset.cache_dir) 
                        if f.startswith("dataset_generation_results_") and f.endswith(".json")]
        
        if not results_files:
            print("No dataset generation results found. Please run generate_fresh_datasets.py first.")
            return
        
        # Sort by timestamp (newest first)
        results_files.sort(reverse=True)
        
        # Load the most recent results
        with open(os.path.join(config.dataset.cache_dir, results_files[0]), 'r') as f:
            dataset_results = json.load(f)
    
    # Filter out datasets with errors
    valid_datasets = [d for d in dataset_results if "error" not in d]
    
    if not valid_datasets:
        print("No valid datasets found. Please run generate_fresh_datasets.py again.")
        return
    
    print(f"Found {len(valid_datasets)} valid datasets to evaluate.")
    
    # Evaluate each dataset
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_info in tqdm(valid_datasets, desc="Evaluating checkpoints"):
        result = evaluate_dataset(config, dataset_info)
        results.append(result)
        print(f"Completed evaluation of {dataset_info['checkpoint_folder']}")
    
    # Save results
    results_file = os.path.join(config.logging.eval_dir, f"checkpoint_evaluation_{timestamp}.json")
    
    # Extract summary metrics
    summary_results = []
    for result in results:
        if "error" in result:
            summary_results.append({
                "checkpoint_folder": result["checkpoint_folder"],
                "checkpoint_num": result.get("checkpoint_num", "unknown"),
                "error": result["error"]
            })
        else:
            summary_results.append({
                "checkpoint_folder": result["checkpoint_folder"],
                "checkpoint_num": result["checkpoint_num"],
                "final_metrics": {k: convert_for_json(v) for k, v in result["final_metrics"].items()},
                "score_difference": convert_for_json(result["score_difference"]),
                "misclassification_rate_base": convert_for_json(result["misclassification_rate_base"]),
                "misclassification_rate_checkpoint": convert_for_json(result["misclassification_rate_checkpoint"]),
                "dataset_analysis": result.get("dataset_analysis", {})
            })
    
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {results_file}")
    
    # Plot comparison of checkpoints
    try:
        # Extract data for plotting
        checkpoint_nums = []
        accuracies = []
        f1_scores = []
        score_diffs = []
        
        for result in results:
            if "error" not in result and "final_metrics" in result:
                checkpoint_nums.append(result["checkpoint_num"])
                accuracies.append(result["final_metrics"].get("accuracy", 0))
                f1_scores.append(result["final_metrics"].get("f1", 0))
                score_diffs.append(result["score_difference"])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        plt.plot(checkpoint_nums, accuracies, 'o-', label='Accuracy')
        plt.plot(checkpoint_nums, f1_scores, 's-', label='F1 Score')
        plt.plot(checkpoint_nums, score_diffs, '^-', label='Score Difference')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Metrics Across Training Epochs - {config.model.checkpoint_repo.split("/")[-1]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        comparison_plot = os.path.join(config.logging.eval_dir, f"checkpoint_comparison_{timestamp}.png")
        plt.savefig(comparison_plot, dpi=300)
        
        print(f"Comparison plot saved to {comparison_plot}")
        
        # Create a summary table
        summary_df = pd.DataFrame({
            "Epoch": checkpoint_nums,
            "Accuracy": accuracies,
            "F1 Score": f1_scores,
            "Score Difference": score_diffs
        })
        
        summary_table = os.path.join(config.logging.eval_dir, f"checkpoint_summary_{timestamp}.csv")
        summary_df.to_csv(summary_table, index=False)
        
        print(f"Summary table saved to {summary_table}")
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
    
    # Log to wandb if configured
    if config.logging.use_wandb:
        try:
            wandb.init(
                project=config.logging.project_name,
                name=f"{config.model.checkpoint_repo.split('/')[-1]}_comparison_{timestamp}",
                config=config.to_dict(),
                reinit=True
            )
            
            # Create a table
            columns = ["Epoch", "Accuracy", "F1 Score", "AUC-ROC", 
                      "Pearson Correlation", "Score Difference"]
            data = []
            
            for result in results:
                if "error" not in result and "final_metrics" in result:
                    data.append([
                        result["checkpoint_num"],
                        float(result["final_metrics"].get("accuracy", 0)),
                        float(result["final_metrics"].get("f1", 0)),
                        float(result["final_metrics"].get("auc_roc", 0)) if "auc_roc" in result["final_metrics"] else 0,
                        float(result["final_metrics"].get("pearson_correlation", 0)) if "pearson_correlation" in result["final_metrics"] else 0,
                        float(result["score_difference"])
                    ])
            
            comparison_table = wandb.Table(columns=columns, data=data)
            wandb.log({"checkpoint_comparison": comparison_table})
            
            # Log the comparison plot
            wandb.log({"checkpoint_comparison_plot": wandb.Image(comparison_plot)})
            
            wandb.finish()
            
        except Exception as e:
            print(f"Error logging to wandb: {e}")


if __name__ == "__main__":
    main() 