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
from huggingface_hub import list_repo_files

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.fresh_config import FreshConfig, get_config_from_args
from data.fresh_dataset import FreshDatasetGenerator
from training.fresh_trainer import FreshRewardModelTrainer


def parse_args():
    """Parse command line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with IRL reward model extraction")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model to compare against (e.g., 'EleutherAI/pythia-70m')")
    parser.add_argument("--checkpoint_repo", type=str, required=True,
                        help="Repository with checkpoints (e.g., 'ajagota71/pythia-70m-detox')")
    parser.add_argument("--checkpoint_pattern", type=str, default="checkpoint-epoch-",
                        help="Pattern to match checkpoint folders")
    parser.add_argument("--specific_checkpoints", type=str, default=None,
                        help="Comma-separated list of specific checkpoint numbers to evaluate")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                        help="Maximum number of checkpoints to evaluate")
    parser.add_argument("--checkpoint_step", type=int, default=1,
                        help="Evaluate every Nth checkpoint")
    
    # Dataset arguments
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation")
    parser.add_argument("--cache_dir", type=str, default="./datasets",
                        help="Directory for caching datasets")
    parser.add_argument("--shared_dataset", action="store_true",
                        help="Use the same prompts for all checkpoints")
    parser.add_argument("--skip_dataset_generation", action="store_true",
                        help="Skip dataset generation and use existing datasets")
    
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
    parser.add_argument("--save_to_drive", action="store_true",
                        help="Save results to Google Drive (for Colab)")
    
    return parser.parse_args()


def get_checkpoint_folders(repo_id, checkpoint_pattern, max_checkpoints=None, 
                          checkpoint_step=1, specific_checkpoints=None):
    """
    Get a list of checkpoint folders from a Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        checkpoint_pattern: Pattern to match checkpoint folders
        max_checkpoints: Maximum number of checkpoints to return
        checkpoint_step: Return every Nth checkpoint
        specific_checkpoints: List of specific checkpoint numbers to return
        
    Returns:
        List of checkpoint folder names
    """
    print(f"Searching for checkpoints in {repo_id}...")
    
    # List all files in the repository
    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing repository files: {e}")
        return []
    
    # Extract unique top-level directories that match the checkpoint pattern
    checkpoint_folders = set()
    for file_path in all_files:
        parts = file_path.split('/')
        if len(parts) > 0 and parts[0].startswith(checkpoint_pattern):
            checkpoint_folders.add(parts[0])
    
    # Convert to list and sort
    checkpoint_folders = sorted(list(checkpoint_folders))
    
    # Extract checkpoint numbers for sorting and filtering
    def get_checkpoint_number(folder):
        try:
            # Try to extract a number from the folder name
            return int(folder.replace(checkpoint_pattern, ""))
        except ValueError:
            # If we can't extract a number, use a large value to sort to the end
            return float('inf')
    
    # Sort by checkpoint number
    checkpoint_folders.sort(key=get_checkpoint_number)
    
    # Filter by specific checkpoints if provided
    if specific_checkpoints:
        specific_nums = [int(x.strip()) for x in specific_checkpoints.split(',')]
        checkpoint_folders = [f for f in checkpoint_folders 
                             if get_checkpoint_number(f) in specific_nums]
    else:
        # Apply checkpoint step
        if checkpoint_step > 1:
            checkpoint_folders = checkpoint_folders[::checkpoint_step]
        
        # Apply max checkpoints
        if max_checkpoints is not None and max_checkpoints > 0:
            checkpoint_folders = checkpoint_folders[:max_checkpoints]
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders: {checkpoint_folders}")
    return checkpoint_folders


def evaluate_checkpoint(config, checkpoint_folder, shared_dataset_generator=None):
    """
    Evaluate a single checkpoint.
    
    Args:
        config: Configuration object
        checkpoint_folder: Checkpoint folder name
        shared_dataset_generator: Optional shared dataset generator
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract checkpoint number from folder name
    try:
        checkpoint_num = int(checkpoint_folder.replace(config.model.checkpoint_pattern, ""))
    except ValueError:
        checkpoint_num = 0
    
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_folder} (Epoch {checkpoint_num})")
    print(f"{'='*80}")
    
    try:
        # Create a unique experiment name for this checkpoint
        experiment_name = f"{config.model.checkpoint_repo.split('/')[-1]}_checkpoint-{checkpoint_num}"
        
        # Initialize wandb if configured
        if config.logging.use_wandb:
            wandb_run = config.init_wandb(experiment_name)
        else:
            wandb_run = None
        
        # Generate or load datasets
        if shared_dataset_generator:
            # Use shared dataset generator
            print(f"Using shared dataset for {checkpoint_folder}...")
            dataset_generator = shared_dataset_generator
            dataset_generator.file_base = f"{checkpoint_num}"
            
            if not dataset_generator.dataset_exists() and not config.dataset.skip_dataset_generation:
                # Generate datasets for this checkpoint
                base_data, checkpoint_data = dataset_generator.generate_datasets(
                    config.model.base_model,
                    config.model.checkpoint_repo,
                    checkpoint_folder
                )
            else:
                # Load existing datasets
                base_data, checkpoint_data = dataset_generator.load_datasets()
        else:
            # Create a new dataset generator for this checkpoint
            dataset_generator = FreshDatasetGenerator(config)
            dataset_generator.file_base = f"{checkpoint_num}"
            
            if not dataset_generator.dataset_exists() and not config.dataset.skip_dataset_generation:
                # Generate datasets for this checkpoint
                base_data, checkpoint_data = dataset_generator.generate_datasets(
                    config.model.base_model,
                    config.model.checkpoint_repo,
                    checkpoint_folder
                )
            else:
                # Load existing datasets
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
        
        # Plot metrics
        print("Generating training metrics plots...")
        plot_paths = trainer.plot_metrics(metrics_history, file_base=f"checkpoint-{checkpoint_num}")
        
        # Analyze scores
        print("Analyzing scores...")
        score_analysis = trainer.analyze_scores(model, data)
        
        # Plot score distribution
        score_dist_plot = trainer.plot_score_distribution(score_analysis, file_base=f"checkpoint-{checkpoint_num}")
        
        # Get final metrics
        final_metrics = metrics_history[-1] if metrics_history else {}
        
        # Log to wandb if configured
        if wandb_run:
            # Log metrics
            wandb.log(final_metrics)
            
            # Log plots
            wandb.log({
                "metrics_plot": wandb.Image(plot_paths["metrics_plot"]),
                "loss_plot": wandb.Image(plot_paths["loss_plot"]),
                "score_distribution": wandb.Image(score_dist_plot)
            })
            
            # Log dataset analysis
            wandb.log({"dataset_analysis": dataset_analysis})
            
            # Log score analysis
            wandb.log({
                "score_difference": score_analysis["score_difference"],
                "base_mean_score": score_analysis["base_mean"],
                "checkpoint_mean_score": score_analysis["checkpoint_mean"],
                "misclassification_rate_base": score_analysis["misclassification_rate_base"],
                "misclassification_rate_checkpoint": score_analysis["misclassification_rate_checkpoint"]
            })
            
            # Finish wandb run
            wandb.finish()
        
        # Return results
        return {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "final_metrics": final_metrics,
            "metrics_history": metrics_history,
            "dataset_analysis": dataset_analysis,
            "score_analysis": score_analysis,
            "score_difference": score_analysis["score_difference"],
            "misclassification_rate_base": score_analysis["misclassification_rate_base"],
            "misclassification_rate_checkpoint": score_analysis["misclassification_rate_checkpoint"],
            "plot_paths": {
                "metrics_plot": plot_paths["metrics_plot"],
                "loss_plot": plot_paths["loss_plot"],
                "score_distribution": score_dist_plot
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error evaluating checkpoint {checkpoint_folder}: {e}")
        return {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "error": str(e)
        }


def main():
    """Main function for checkpoint evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = FreshConfig.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Set up directories
    config.setup_directories()
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get checkpoint folders
    checkpoint_folders = get_checkpoint_folders(
        config.model.checkpoint_repo,
        config.model.checkpoint_pattern,
        max_checkpoints=config.model.max_checkpoints,
        checkpoint_step=config.model.checkpoint_step,
        specific_checkpoints=config.model.specific_checkpoints
    )
    
    if not checkpoint_folders:
        print(f"No checkpoint folders found in {config.model.checkpoint_repo}")
        return
    
    # Initialize shared dataset generator if using shared dataset
    shared_dataset_generator = None
    if config.dataset.shared_dataset:
        print("\n=== Generating shared dataset ===")
        shared_dataset_generator = FreshDatasetGenerator(config)
        
        # Use the first checkpoint for the shared dataset
        first_checkpoint = checkpoint_folders[0]
        shared_dataset_generator.file_base = "shared"
        
        if not shared_dataset_generator.dataset_exists() and not config.dataset.skip_dataset_generation:
            # Generate datasets
            base_data, checkpoint_data = shared_dataset_generator.generate_datasets(
                config.model.base_model,
                config.model.checkpoint_repo,
                first_checkpoint
            )
        else:
            # Load existing datasets
            base_data, checkpoint_data = shared_dataset_generator.load_datasets()
        
        print(f"Shared dataset generation complete")
    
    # Evaluate checkpoints
    results = []
    for checkpoint_folder in tqdm(checkpoint_folders, desc="Evaluating checkpoints"):
        result = evaluate_checkpoint(
            config,
            checkpoint_folder,
            shared_dataset_generator
        )
        results.append(result)
        print(f"Completed evaluation of {checkpoint_folder}")
    
    # Save results
    results_file = os.path.join(config.logging.eval_dir, f"checkpoint_evaluation_{timestamp}.json")
    
    # Convert numpy values to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
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
                config=config.to_dict()
            )
            
            # Create a table
            columns = ["Epoch", "Accuracy", "F1 Score", "AUC-ROC", 
                      "Pearson Correlation", "Score Difference"]
            data = []
            
            for result in results:
                if "error" not in result and "final_metrics" in result:
                    data.append([
                        result["checkpoint_num"],
                        result["final_metrics"].get("accuracy", 0),
                        result["final_metrics"].get("f1", 0),
                        result["final_metrics"].get("auc_roc", 0),
                        result["final_metrics"].get("pearson_correlation", 0),
                        result["score_difference"]
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