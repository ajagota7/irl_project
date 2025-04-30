# scripts/evaluate_huggingface_checkpoints.py
import argparse
import os
import torch
import wandb
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import list_repo_files, hf_hub_download

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics, analyze_score_distribution, analyze_errors
from utils.logging import setup_wandb


def parse_args():
    """Parse command line arguments for finetuned checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned checkpoints with IRL")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--username", type=str, default="ajagota71",
                        help="Hugging Face username")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., 'pythia-70m-detox')")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name (e.g., 'EleutherAI/pythia-70m')")
    parser.add_argument("--checkpoint_pattern", type=str, default="checkpoint-epoch-",
                        help="Pattern to match checkpoint folders")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                        help="Maximum number of checkpoints to evaluate (None for all)")
    parser.add_argument("--checkpoint_step", type=int, default=1,
                        help="Evaluate every Nth checkpoint")
    parser.add_argument("--specific_checkpoints", type=str, default=None,
                        help="Comma-separated list of specific checkpoint numbers to evaluate")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default="./datasets", 
                        help="Directory with datasets")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate for each checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--unfrozen_layers", type=int, default=1, 
                        help="Number of unfrozen layers in the model")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoint_evaluation", 
                        help="Output directory for evaluation results")
    parser.add_argument("--project_name", type=str, default="irl-checkpoint-evaluation", 
                        help="Project name for wandb")
    parser.add_argument("--experiment_name", type=str, 
                        help="Experiment name for wandb")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use wandb for logging")
    parser.add_argument("--save_to_drive", action="store_true", 
                        help="Save results to Google Drive (for Colab)")
    
    return parser.parse_args()


def get_checkpoint_folders(username, model_name, checkpoint_pattern="checkpoint-epoch-", 
                          max_checkpoints=None, checkpoint_step=1, specific_checkpoints=None):
    """
    Get a list of checkpoint folders from a Hugging Face repository.
    
    Args:
        username: Hugging Face username
        model_name: Model name
        checkpoint_pattern: Pattern to match checkpoint folders
        max_checkpoints: Maximum number of checkpoints to return
        checkpoint_step: Return every Nth checkpoint
        specific_checkpoints: List of specific checkpoint numbers to return
        
    Returns:
        List of checkpoint folder names
    """
    repo_id = f"{username}/{model_name}"
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
            return float('inf')  # Put folders without numbers at the end
    
    # Sort by checkpoint number
    checkpoint_folders.sort(key=get_checkpoint_number)
    
    # Filter specific checkpoints if requested
    if specific_checkpoints:
        specific_nums = [int(x.strip()) for x in specific_checkpoints.split(',')]
        checkpoint_folders = [f for f in checkpoint_folders 
                             if get_checkpoint_number(f) in specific_nums]
    else:
        # Apply step and max_checkpoints
        checkpoint_folders = checkpoint_folders[::checkpoint_step]
        if max_checkpoints:
            checkpoint_folders = checkpoint_folders[:max_checkpoints]
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders:")
    for folder in checkpoint_folders:
        print(f"  - {folder}")
    
    return checkpoint_folders


def evaluate_checkpoint(username, model_name, checkpoint_folder, base_model, config):
    """
    Evaluate a single checkpoint.
    
    Args:
        username: Hugging Face username
        model_name: Model name
        checkpoint_folder: Checkpoint folder name
        base_model: Base model name
        config: Configuration object
        
    Returns:
        Dictionary with evaluation results
    """
    repo_id = f"{username}/{model_name}"
    checkpoint_path = f"{repo_id}/{checkpoint_folder}"
    
    # Extract checkpoint number for naming
    checkpoint_num = "unknown"
    if "-epoch-" in checkpoint_folder:
        checkpoint_num = checkpoint_folder.split("-epoch-")[1]
    
    print(f"\n=== Evaluating checkpoint {checkpoint_folder} (epoch {checkpoint_num}) ===")
    
    # Create a unique experiment name
    experiment_name = f"{model_name}_epoch-{checkpoint_num}_evaluation"
    
    # Create a copy of the config for this checkpoint
    checkpoint_config = Config.from_dict(config.to_dict())
    checkpoint_config.logging.experiment_name = experiment_name
    
    # Initialize wandb for this checkpoint
    wandb_run = None
    if checkpoint_config.logging.use_wandb:
        wandb_run = wandb.init(
            project=checkpoint_config.logging.project_name,
            name=experiment_name,
            config=checkpoint_config.to_dict(),
            mode=checkpoint_config.logging.wandb_mode,
            reinit=True
        )
        wandb.config.update({
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "base_model": base_model
        })
    
    try:
        # Initialize dataset generator
        dataset_generator = DatasetGenerator(checkpoint_config)
        
        # Set file base for this checkpoint
        dataset_generator.file_base = f"{model_name}_epoch-{checkpoint_num}"
        
        # Override model paths to use the checkpoint and base model
        dataset_generator.original_model_path = checkpoint_path
        dataset_generator.detoxified_model_path = base_model
        
        # Generate datasets
        print(f"Generating datasets from {checkpoint_path}...")
        original_data, detoxified_data = dataset_generator.generate_datasets()
        
        # Analyze datasets
        print("Analyzing datasets...")
        analysis = dataset_generator.analyze_datasets()
        
        # Log dataset analysis to wandb
        if wandb_run is not None:
            wandb.log({
                "dataset_generation_complete": True,
                "dataset_analysis": analysis
            })
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = RewardModelTrainer(checkpoint_config)
        
        # Prepare data
        train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
        
        # Train model
        print(f"Training reward model...")
        reward_model, metrics_history = trainer.train(train_data, test_data)
        
        # Plot metrics
        print("Generating training metrics plots...")
        metadata = plot_metrics(metrics_history, checkpoint_config, file_base=dataset_generator.file_base)
        
        # Analyze score distribution
        print("Analyzing score distribution...")
        original_scores, detoxified_scores = analyze_score_distribution(
            reward_model, trainer.tokenizer, test_data, checkpoint_config
        )
        
        # Analyze errors
        print("Analyzing errors...")
        misclassified_original, misclassified_detoxified = analyze_errors(
            reward_model, trainer.tokenizer, test_data, checkpoint_config
        )
        
        # Compile results
        results = {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "metrics_history": metrics_history,
            "final_metrics": metrics_history[-1] if metrics_history else {},
            "original_scores_mean": float(np.mean(original_scores)),
            "detoxified_scores_mean": float(np.mean(detoxified_scores)),
            "score_difference": float(np.mean(detoxified_scores) - np.mean(original_scores)),
            "misclassification_rate_original": len(misclassified_original) / len(test_data['original']),
            "misclassification_rate_detoxified": len(misclassified_detoxified) / len(test_data['detoxified']),
            "dataset_analysis": analysis
        }
        
        # Log final results to wandb
        if wandb_run is not None:
            wandb.log({
                "evaluation_complete": True,
                "final_metrics": results["final_metrics"],
                "score_difference": results["score_difference"],
                "misclassification_rate_original": results["misclassification_rate_original"],
                "misclassification_rate_detoxified": results["misclassification_rate_detoxified"],
            })
            
            # Finish wandb run
            wandb.finish()
        
        return results
    
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_folder}: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error to wandb
        if wandb_run is not None:
            wandb.log({"evaluation_error": str(e)})
            wandb.finish()
        
        # Return error
        return {
            "checkpoint_folder": checkpoint_folder,
            "checkpoint_num": checkpoint_num,
            "error": str(e)
        }


def evaluate_checkpoints():
    """Evaluate multiple checkpoint models."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update config with command line arguments
    if args.output_dir:
        config.logging.eval_dir = args.output_dir
    if args.dataset_dir:
        config.dataset.cache_dir = args.dataset_dir
    if args.num_samples:
        config.dataset.num_samples = args.num_samples
    if args.max_new_tokens:
        config.dataset.max_new_tokens = args.max_new_tokens
    if args.temperature:
        config.dataset.temperature = args.temperature
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.project_name:
        config.logging.project_name = args.project_name
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.use_wandb:
        config.logging.use_wandb = True
    if args.save_to_drive:
        config.logging.save_to_drive = True
    
    # Set random seeds
    torch.manual_seed(config.training.seed)
    import numpy as np
    np.random.seed(config.training.seed)
    
    # Set up directories
    config.setup_directories()
    
    # Get checkpoint folders
    checkpoint_folders = get_checkpoint_folders(
        args.username,
        args.model_name,
        checkpoint_pattern=args.checkpoint_pattern,
        max_checkpoints=args.max_checkpoints,
        checkpoint_step=args.checkpoint_step,
        specific_checkpoints=args.specific_checkpoints
    )
    
    if not checkpoint_folders:
        print(f"No checkpoint folders found for {args.username}/{args.model_name}")
        return
    
    # Evaluate each checkpoint
    results = []
    for checkpoint_folder in tqdm(checkpoint_folders, desc="Evaluating checkpoints"):
        result = evaluate_checkpoint(
            args.username, 
            args.model_name, 
            checkpoint_folder, 
            args.base_model,
            config
        )
        results.append(result)
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.logging.eval_dir, f"{args.model_name}_evaluation_results_{timestamp}.json")
    
    # Convert numpy values to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
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
                "misclassification_rate_original": convert_for_json(result["misclassification_rate_original"]),
                "misclassification_rate_detoxified": convert_for_json(result["misclassification_rate_detoxified"]),
                "dataset_analysis": result.get("dataset_analysis", {})
            })
    
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {results_file}")
    
    # Plot comparison of checkpoints
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
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
        plt.title(f'Metrics Across Training Epochs - {args.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        comparison_plot = os.path.join(config.logging.eval_dir, f"{args.model_name}_comparison_{timestamp}.png")
        plt.savefig(comparison_plot, dpi=300)
        
        print(f"Comparison plot saved to {comparison_plot}")
        
        # Create a summary table
        summary_df = pd.DataFrame({
            "Epoch": checkpoint_nums,
            "Accuracy": accuracies,
            "F1 Score": f1_scores,
            "Score Difference": score_diffs
        })
        
        summary_table = os.path.join(config.logging.eval_dir, f"{args.model_name}_summary_{timestamp}.csv")
        summary_df.to_csv(summary_table, index=False)
        
        print(f"Summary table saved to {summary_table}")
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
    
    # Log to wandb if configured
    if config.logging.use_wandb:
        try:
            wandb.init(
                project=config.logging.project_name,
                name=f"{args.model_name}_comparison_{timestamp}",
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
    evaluate_checkpoints() 