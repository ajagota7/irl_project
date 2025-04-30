import argparse
import os
import torch
import wandb
from huggingface_hub import list_models, model_info
from tqdm import tqdm

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator
from training.trainer import RewardModelTrainer
from training.evaluation import plot_metrics, analyze_score_distribution, analyze_errors
from utils.logging import setup_wandb


def parse_args():
    """Parse command line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints with IRL")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name on Hugging Face (e.g., 'username/model-name')")
    parser.add_argument("--checkpoint_pattern", type=str, default="checkpoint-",
                        help="Pattern to match checkpoint names")
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
    
    return parser.parse_args()


def get_checkpoint_models(base_model, checkpoint_pattern="checkpoint-", max_checkpoints=None, 
                          checkpoint_step=1, specific_checkpoints=None):
    """
    Get a list of checkpoint models from Hugging Face.
    
    Args:
        base_model: Base model name on Hugging Face
        checkpoint_pattern: Pattern to match checkpoint names
        max_checkpoints: Maximum number of checkpoints to return
        checkpoint_step: Return every Nth checkpoint
        specific_checkpoints: List of specific checkpoint numbers to return
        
    Returns:
        List of checkpoint model names
    """
    print(f"Searching for checkpoints of {base_model}...")
    
    # Get all models from the user
    user = base_model.split('/')[0]
    models = list_models(author=user)
    
    # Filter models that match the base model and checkpoint pattern
    checkpoint_models = []
    for model in models:
        model_id = model.modelId
        if model_id.startswith(base_model) and checkpoint_pattern in model_id:
            checkpoint_models.append(model_id)
    
    # Sort checkpoints by number
    def get_checkpoint_number(model_id):
        try:
            # Extract the number after the checkpoint pattern
            parts = model_id.split(checkpoint_pattern)
            if len(parts) > 1:
                return int(parts[1])
            return float('inf')  # Put models without numbers at the end
        except ValueError:
            return float('inf')
    
    checkpoint_models.sort(key=get_checkpoint_number)
    
    # Filter specific checkpoints if requested
    if specific_checkpoints:
        specific_nums = [int(x.strip()) for x in specific_checkpoints.split(',')]
        checkpoint_models = [m for m in checkpoint_models 
                            if get_checkpoint_number(m) in specific_nums]
    else:
        # Apply step and max_checkpoints
        checkpoint_models = checkpoint_models[::checkpoint_step]
        if max_checkpoints:
            checkpoint_models = checkpoint_models[:max_checkpoints]
    
    print(f"Found {len(checkpoint_models)} checkpoint models:")
    for model in checkpoint_models:
        print(f"  - {model}")
    
    return checkpoint_models


def evaluate_checkpoint(checkpoint_model, config, dataset_generator=None):
    """
    Evaluate a single checkpoint model.
    
    Args:
        checkpoint_model: Checkpoint model name
        config: Configuration object
        dataset_generator: Optional dataset generator to reuse
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_model}")
    print(f"{'='*80}")
    
    # Update config for this checkpoint
    checkpoint_config = Config.from_dict(config.to_dict())
    
    # Extract checkpoint number for naming
    checkpoint_num = "unknown"
    if "-checkpoint-" in checkpoint_model:
        checkpoint_num = checkpoint_model.split("-checkpoint-")[1]
    
    # Update experiment name
    if checkpoint_config.logging.experiment_name:
        checkpoint_config.logging.experiment_name = f"{checkpoint_config.logging.experiment_name}_checkpoint-{checkpoint_num}"
    else:
        checkpoint_config.logging.experiment_name = f"checkpoint-{checkpoint_num}_evaluation"
    
    # Initialize wandb for this checkpoint
    if checkpoint_config.logging.use_wandb:
        wandb_run = setup_wandb(checkpoint_config)
        wandb.config.update({"checkpoint_model": checkpoint_model})
    else:
        wandb_run = None
    
    try:
        # Generate datasets if not provided
        if dataset_generator is None:
            print("Initializing dataset generator...")
            dataset_generator = DatasetGenerator(checkpoint_config)
            
            # Override model paths to use the checkpoint model
            dataset_generator.original_model_path = checkpoint_model
            
            # Generate datasets
            print(f"Generating datasets for {checkpoint_model}...")
            original_data, detoxified_data = dataset_generator.generate_datasets()
        else:
            # Load existing datasets
            print("Loading existing datasets...")
            original_data, detoxified_data = dataset_generator.load_datasets()
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = RewardModelTrainer(checkpoint_config)
        
        # Prepare data
        train_data, test_data = trainer.prepare_data(original_data, detoxified_data)
        
        # Train model
        print(f"Training reward model for checkpoint {checkpoint_num}...")
        reward_model, metrics_history = trainer.train(train_data, test_data)
        
        # Plot metrics
        print("Generating training metrics plots...")
        metadata = plot_metrics(metrics_history, checkpoint_config, file_base=f"checkpoint-{checkpoint_num}")
        
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
            "checkpoint_model": checkpoint_model,
            "checkpoint_num": checkpoint_num,
            "metrics_history": metrics_history,
            "final_metrics": metrics_history[-1] if metrics_history else {},
            "original_scores_mean": float(np.mean(original_scores)),
            "detoxified_scores_mean": float(np.mean(detoxified_scores)),
            "score_difference": float(np.mean(detoxified_scores) - np.mean(original_scores)),
            "misclassification_rate_original": len(misclassified_original) / len(test_data['original']),
            "misclassification_rate_detoxified": len(misclassified_detoxified) / len(test_data['detoxified']),
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
        print(f"Error evaluating checkpoint {checkpoint_model}: {e}")
        if wandb_run is not None:
            wandb.finish()
        return {"checkpoint_model": checkpoint_model, "error": str(e)}


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
    
    # Get checkpoint models
    checkpoint_models = get_checkpoint_models(
        args.base_model,
        checkpoint_pattern=args.checkpoint_pattern,
        max_checkpoints=args.max_checkpoints,
        checkpoint_step=args.checkpoint_step,
        specific_checkpoints=args.specific_checkpoints
    )
    
    if not checkpoint_models:
        print(f"No checkpoint models found for {args.base_model}")
        return
    
    # Initialize dataset generator if using the same dataset for all checkpoints
    dataset_generator = None
    if args.dataset_file_base:
        print(f"Using existing dataset: {args.dataset_file_base}")
        dataset_generator = DatasetGenerator(config)
        dataset_generator.file_base = args.dataset_file_base
    
    # Evaluate each checkpoint
    results = []
    for checkpoint_model in tqdm(checkpoint_models, desc="Evaluating checkpoints"):
        result = evaluate_checkpoint(checkpoint_model, config, dataset_generator)
        results.append(result)
    
    # Save overall results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.logging.eval_dir, f"checkpoint_evaluation_results_{timestamp}.json")
    
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
                "checkpoint_model": result["checkpoint_model"],
                "error": result["error"]
            })
        else:
            summary_results.append({
                "checkpoint_model": result["checkpoint_model"],
                "checkpoint_num": result["checkpoint_num"],
                "final_metrics": {k: convert_for_json(v) for k, v in result["final_metrics"].items()},
                "score_difference": convert_for_json(result["score_difference"]),
                "misclassification_rate_original": convert_for_json(result["misclassification_rate_original"]),
                "misclassification_rate_detoxified": convert_for_json(result["misclassification_rate_detoxified"]),
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