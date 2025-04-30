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
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face repository ID (e.g., 'username/model-name')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory in the repo containing checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                        help="Maximum number of checkpoints to evaluate (None for all)")
    parser.add_argument("--checkpoint_step", type=int, default=1,
                        help="Evaluate every Nth checkpoint")
    parser.add_argument("--specific_checkpoints", type=str, default=None,
                        help="Comma-separated list of specific checkpoint numbers to evaluate")
    
    # Dataset arguments
    parser.add_argument("--dataset_dir", type=str, default="./datasets", 
                        help="Directory with datasets")
    parser.add_argument("--dataset_file_base", type=str, 
                        help="Base filename for dataset files")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate for each checkpoint")
    
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
    
    # Pipeline control
    parser.add_argument("--skip_dataset_generation", action="store_true",
                        help="Skip dataset generation and use existing datasets")
    parser.add_argument("--shared_dataset", action="store_true",
                        help="Use a single dataset for all checkpoints (generated from the first checkpoint)")
    
    return parser.parse_args()


def get_finetuned_checkpoints(repo_id, checkpoint_dir="checkpoints", max_checkpoints=None, 
                             checkpoint_step=1, specific_checkpoints=None):
    """
    Get a list of finetuned checkpoint paths from a Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        checkpoint_dir: Directory in the repo containing checkpoints
        max_checkpoints: Maximum number of checkpoints to return
        checkpoint_step: Return every Nth checkpoint
        specific_checkpoints: List of specific checkpoint numbers to return
        
    Returns:
        List of checkpoint paths
    """
    print(f"Searching for checkpoints in {repo_id}/{checkpoint_dir}...")
    
    # List all files in the repository
    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing repository files: {e}")
        return []
    
    # Filter files to find checkpoint directories
    checkpoint_dirs = set()
    for file_path in all_files:
        if file_path.startswith(checkpoint_dir):
            # Extract the checkpoint directory
            parts = file_path.split('/')
            if len(parts) > 1:
                checkpoint_dirs.add('/'.join(parts[:2]))
    
    # Convert to list and sort
    checkpoint_paths = sorted(list(checkpoint_dirs))
    
    # Extract checkpoint numbers for sorting and filtering
    def get_checkpoint_number(path):
        try:
            # Try to extract a number from the path
            parts = path.split('-')
            if len(parts) > 1:
                return int(parts[-1])
            return float('inf')  # Put paths without numbers at the end
        except ValueError:
            return float('inf')
    
    # Sort by checkpoint number
    checkpoint_paths.sort(key=get_checkpoint_number)
    
    # Filter specific checkpoints if requested
    if specific_checkpoints:
        specific_nums = [int(x.strip()) for x in specific_checkpoints.split(',')]
        checkpoint_paths = [p for p in checkpoint_paths 
                           if get_checkpoint_number(p) in specific_nums]
    else:
        # Apply step and max_checkpoints
        checkpoint_paths = checkpoint_paths[::checkpoint_step]
        if max_checkpoints:
            checkpoint_paths = checkpoint_paths[:max_checkpoints]
    
    print(f"Found {len(checkpoint_paths)} checkpoint directories:")
    for path in checkpoint_paths:
        print(f"  - {path}")
    
    return checkpoint_paths


def evaluate_finetuned_checkpoint(repo_id, checkpoint_path, config, dataset_generator=None):
    """
    Evaluate a single finetuned checkpoint.
    
    Args:
        repo_id: Hugging Face repository ID
        checkpoint_path: Path to the checkpoint directory in the repo
        config: Configuration object
        dataset_generator: Optional dataset generator to reuse
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Update config for this checkpoint
    checkpoint_config = Config.from_dict(config.to_dict())
    
    # Extract checkpoint number for naming
    checkpoint_num = "unknown"
    if "-" in checkpoint_path:
        checkpoint_num = checkpoint_path.split("-")[-1]
    
    # Update experiment name
    if checkpoint_config.logging.experiment_name:
        checkpoint_config.logging.experiment_name = f"{checkpoint_config.logging.experiment_name}_checkpoint-{checkpoint_num}"
    else:
        checkpoint_config.logging.experiment_name = f"checkpoint-{checkpoint_num}_evaluation"
    
    # Initialize wandb for this checkpoint
    if checkpoint_config.logging.use_wandb:
        wandb_run = setup_wandb(checkpoint_config)
        wandb.config.update({"checkpoint_path": checkpoint_path})
    else:
        wandb_run = None
    
    try:
        # Load datasets
        if dataset_generator is None:
            # Initialize dataset generator
            from data.dataset import DatasetGenerator
            dataset_generator = DatasetGenerator(checkpoint_config)
            
            # Set file base for this checkpoint
            dataset_generator.file_base = f"checkpoint-{checkpoint_num}"
            
            # Override model path to use the checkpoint
            dataset_generator.original_model_path = f"{repo_id}/{checkpoint_path}"
            
            # Generate or load datasets
            if not dataset_generator.dataset_exists() and not config.dataset.use_cached:
                print(f"Generating datasets for {checkpoint_path}...")
                original_data, detoxified_data = dataset_generator.generate_datasets()
            else:
                print(f"Loading existing datasets for {checkpoint_path}...")
                original_data, detoxified_data = dataset_generator.load_datasets()
        else:
            # Use provided dataset generator
            print(f"Using provided datasets...")
            original_data, detoxified_data = dataset_generator.load_datasets()
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = RewardModelTrainer(checkpoint_config)
        
        # Prepare data
        train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
        
        # Train model
        print(f"Training reward model for {checkpoint_path}...")
        reward_model, metrics_history = trainer.train(train_data, test_data)
        
        # Plot metrics
        print("Generating training metrics plots...")
        plot_metrics(metrics_history, checkpoint_config, file_base=dataset_generator.file_base)
        
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
        
        # Log final metrics
        if wandb_run is not None:
            wandb.log({
                "evaluation_complete": True,
                "final_metrics": metrics_history[-1],
                "total_epochs": len(metrics_history),
                "score_difference": np.mean(detoxified_scores) - np.mean(original_scores),
                "misclassification_rate_original": len(misclassified_original) / len(original_scores),
                "misclassification_rate_detoxified": len(misclassified_detoxified) / len(detoxified_scores)
            })
            
            # Finish wandb run
            wandb.finish()
        
        # Return results
        return {
            "checkpoint_path": checkpoint_path,
            "checkpoint_num": checkpoint_num,
            "final_metrics": metrics_history[-1],
            "score_difference": np.mean(detoxified_scores) - np.mean(original_scores),
            "misclassification_rate_original": len(misclassified_original) / len(original_scores),
            "misclassification_rate_detoxified": len(misclassified_detoxified) / len(detoxified_scores)
        }
        
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error to wandb
        if wandb_run is not None:
            wandb.log({"evaluation_error": str(e)})
            wandb.finish()
        
        # Return error
        return {
            "checkpoint_path": checkpoint_path,
            "checkpoint_num": checkpoint_num,
            "error": str(e)
        }


def evaluate_finetuned_checkpoints():
    """Evaluate multiple finetuned checkpoints."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update config with command line arguments
    if args.dataset_dir:
        config.dataset.cache_dir = args.dataset_dir
    if args.output_dir:
        config.logging.eval_dir = args.output_dir
    if args.num_samples:
        config.dataset.num_samples = args.num_samples
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
    
    # Get checkpoint paths
    checkpoint_paths = get_finetuned_checkpoints(
        args.repo_id,
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=args.max_checkpoints,
        checkpoint_step=args.checkpoint_step,
        specific_checkpoints=args.specific_checkpoints
    )
    
    if not checkpoint_paths:
        print(f"No checkpoint paths found in {args.repo_id}/{args.checkpoint_dir}")
        return
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize shared dataset generator if using shared dataset
    shared_dataset_generator = None
    if args.shared_dataset and not args.skip_dataset_generation:
        print("\n=== Generating shared dataset from first checkpoint ===")
        # Use the first checkpoint to generate a shared dataset
        first_checkpoint = checkpoint_paths[0]
        
        # Extract checkpoint number for naming
        checkpoint_num = "unknown"
        if "-" in first_checkpoint:
            checkpoint_num = first_checkpoint.split("-")[-1]
        
        # Create dataset config
        dataset_config = Config.from_dict(config.to_dict())
        dataset_config.logging.experiment_name = f"shared_dataset_from_checkpoint-{checkpoint_num}"
        
        # Initialize wandb for dataset generation
        if dataset_config.logging.use_wandb:
            wandb_run = wandb.init(
                project=dataset_config.logging.project_name,
                name=dataset_config.logging.experiment_name,
                config=dataset_config.to_dict(),
                mode=dataset_config.logging.wandb_mode
            )
            wandb.config.update({"checkpoint_path": first_checkpoint})
        else:
            wandb_run = None
        
        # Initialize dataset generator
        from data.dataset import DatasetGenerator
        shared_dataset_generator = DatasetGenerator(dataset_config)
        
        # Override model path to use the checkpoint
        shared_dataset_generator.original_model_path = f"{args.repo_id}/{first_checkpoint}"
        
        # Set a unique file base for this shared dataset
        shared_dataset_generator.file_base = f"shared_dataset_from_checkpoint-{checkpoint_num}"
        
        # Generate datasets
        print(f"Generating shared dataset from {first_checkpoint}...")
        original_data, detoxified_data = shared_dataset_generator.generate_datasets()
        
        # Analyze datasets
        print("Analyzing shared dataset...")
        analysis = shared_dataset_generator.analyze_datasets()
        
        # Log to wandb if configured
        if wandb_run is not None:
            wandb.log({
                "dataset_generation_complete": True,
                "dataset_analysis": analysis
            })
            
            # Log the datasets as artifacts
            dataset_artifact = wandb.Artifact(
                f"shared_dataset_from_checkpoint-{checkpoint_num}", 
                type="dataset"
            )
            dataset_artifact.add_dir(dataset_config.dataset.cache_dir)
            wandb.log_artifact(dataset_artifact)
            
            # Finish wandb run
            wandb.finish()
        
        print(f"Shared dataset generation complete")
    
    # Evaluate checkpoints
    print("\n=== Evaluating checkpoints ===")
    evaluation_results = []
    
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        try:
            # Determine which dataset generator to use
            dataset_generator = None
            if args.shared_dataset:
                # Use the shared dataset
                dataset_generator = shared_dataset_generator
            
            # Evaluate the checkpoint
            result = evaluate_finetuned_checkpoint(args.repo_id, checkpoint_path, config, dataset_generator)
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            evaluation_results.append({
                "checkpoint_path": checkpoint_path,
                "error": str(e)
            })
    
    # Save evaluation results
    evaluation_results_file = os.path.join(config.logging.eval_dir, f"checkpoint_evaluation_results_{timestamp}.json")
    
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
    for result in evaluation_results:
        if "error" in result:
            summary_results.append({
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_num": result.get("checkpoint_num", "unknown"),
                "error": result["error"]
            })
        else:
            summary_results.append({
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_num": result["checkpoint_num"],
                "final_metrics": {k: convert_for_json(v) for k, v in result["final_metrics"].items()},
                "score_difference": convert_for_json(result["score_difference"]),
                "misclassification_rate_original": convert_for_json(result["misclassification_rate_original"]),
                "misclassification_rate_detoxified": convert_for_json(result["misclassification_rate_detoxified"]),
            })
    
    with open(evaluation_results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {evaluation_results_file}")
    
    # Plot comparison of checkpoints
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Extract data for plotting
        checkpoint_nums = []
        accuracies = []
        f1_scores = []
        score_diffs = []
        
        for result in evaluation_results:
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
        
        plt.xlabel('Checkpoint')
        plt.ylabel('Score')
        plt.title('Metrics Across Checkpoints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        comparison_plot = os.path.join(config.logging.eval_dir, f"checkpoint_comparison_{timestamp}.png")
        plt.savefig(comparison_plot, dpi=300)
        
        print(f"Comparison plot saved to {comparison_plot}")
        
        # Create a summary table
        summary_df = pd.DataFrame({
            "Checkpoint": checkpoint_nums,
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
                name=f"checkpoint_comparison_{timestamp}",
                config=config.to_dict()
            )
            
            # Create a table
            columns = ["Checkpoint", "Accuracy", "F1 Score", "AUC-ROC", 
                      "Pearson Correlation", "Score Difference"]
            data = []
            
            for result in evaluation_results:
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
    evaluate_finetuned_checkpoints() 