import argparse
import os
import torch
import wandb
import json
from datetime import datetime
from tqdm import tqdm

from config.config import Config, get_config_from_args
from scripts.generate_datasets_from_checkpoints import get_checkpoint_models, generate_datasets_from_checkpoints
from scripts.evaluate_checkpoints import evaluate_checkpoint


def parse_args():
    """Parse command line arguments for the complete evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run complete checkpoint evaluation pipeline")
    
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
    parser.add_argument("--dataset_dir", type=str, default="./datasets", 
                        help="Output directory for datasets")
    
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
    parser.add_argument("--output_dir", type=str, default="./checkpoint_evaluation", 
                        help="Output directory for evaluation results")
    parser.add_argument("--project_name", type=str, default="irl-checkpoint-evaluation", 
                        help="Project name for wandb")
    parser.add_argument("--experiment_name", type=str, 
                        help="Base experiment name for wandb")
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


def run_evaluation_pipeline():
    """Run the complete checkpoint evaluation pipeline."""
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
    if args.max_new_tokens:
        config.dataset.max_new_tokens = args.max_new_tokens
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.temperature:
        config.dataset.temperature = args.temperature
    if args.top_p:
        config.dataset.top_p = args.top_p
    if args.epochs:
        config.training.epochs = args.epochs
    if args.train_batch_size:
        config.training.batch_size = args.train_batch_size
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
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize shared dataset generator if using shared dataset
    shared_dataset_generator = None
    
    # Generate datasets if needed
    if not args.skip_dataset_generation:
        if args.shared_dataset:
            print("\n=== Generating shared dataset from first checkpoint ===")
            # Use the first checkpoint to generate a shared dataset
            first_checkpoint = checkpoint_models[0]
            
            # Extract checkpoint number for naming
            checkpoint_num = "unknown"
            if "-checkpoint-" in first_checkpoint:
                checkpoint_num = first_checkpoint.split("-checkpoint-")[1]
            
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
                wandb.config.update({"checkpoint_model": first_checkpoint})
            else:
                wandb_run = None
            
            # Initialize dataset generator
            from data.dataset import DatasetGenerator
            shared_dataset_generator = DatasetGenerator(dataset_config)
            
            # Override model paths to use the checkpoint model
            shared_dataset_generator.original_model_path = first_checkpoint
            
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
        else:
            print("\n=== Generating individual datasets for each checkpoint ===")
            # Generate individual datasets for each checkpoint
            dataset_results = []
            
            for checkpoint_model in tqdm(checkpoint_models, desc="Generating datasets"):
                try:
                    # Extract checkpoint number for naming
                    checkpoint_num = "unknown"
                    if "-checkpoint-" in checkpoint_model:
                        checkpoint_num = checkpoint_model.split("-checkpoint-")[1]
                    
                    # Create dataset config
                    dataset_config = Config.from_dict(config.to_dict())
                    dataset_config.logging.experiment_name = f"dataset_checkpoint-{checkpoint_num}"
                    
                    # Initialize wandb for this checkpoint
                    if dataset_config.logging.use_wandb:
                        wandb_run = wandb.init(
                            project=dataset_config.logging.project_name,
                            name=dataset_config.logging.experiment_name,
                            config=dataset_config.to_dict(),
                            mode=dataset_config.logging.wandb_mode,
                            reinit=True
                        )
                        wandb.config.update({"checkpoint_model": checkpoint_model})
                    else:
                        wandb_run = None
                    
                    # Initialize dataset generator
                    from data.dataset import DatasetGenerator
                    generator = DatasetGenerator(dataset_config)
                    
                    # Override model paths to use the checkpoint model
                    generator.original_model_path = checkpoint_model
                    
                    # Set a unique file base for this checkpoint
                    generator.file_base = f"checkpoint-{checkpoint_num}"
                    
                    # Generate datasets
                    print(f"Generating datasets for {checkpoint_model}...")
                    original_data, detoxified_data = generator.generate_datasets()
                    
                    # Analyze datasets
                    print("Analyzing datasets...")
                    analysis = generator.analyze_datasets()
                    
                    # Log to wandb if configured
                    if wandb_run is not None:
                        wandb.log({
                            "dataset_generation_complete": True,
                            "dataset_analysis": analysis
                        })
                        
                        # Log the datasets as artifacts
                        dataset_artifact = wandb.Artifact(
                            f"datasets_checkpoint-{checkpoint_num}", 
                            type="dataset"
                        )
                        dataset_artifact.add_dir(dataset_config.dataset.cache_dir)
                        wandb.log_artifact(dataset_artifact)
                        
                        # Finish wandb run
                        wandb.finish()
                    
                    # Record results
                    dataset_results.append({
                        "checkpoint_model": checkpoint_model,
                        "checkpoint_num": checkpoint_num,
                        "file_base": generator.file_base,
                        "analysis": analysis
                    })
                    
                    print(f"Dataset generation complete for {checkpoint_model}")
                    
                except Exception as e:
                    print(f"Error generating datasets for {checkpoint_model}: {e}")
                    dataset_results.append({
                        "checkpoint_model": checkpoint_model,
                        "error": str(e)
                    })
            
            # Save dataset generation results
            dataset_results_file = os.path.join(config.dataset.cache_dir, f"dataset_generation_results_{timestamp}.json")
            with open(dataset_results_file, 'w') as f:
                json.dump(dataset_results, f, indent=2)
            
            print(f"Dataset generation complete. Results saved to {dataset_results_file}")
    
    # Evaluate checkpoints
    print("\n=== Evaluating checkpoints ===")
    evaluation_results = []
    
    for checkpoint_model in tqdm(checkpoint_models, desc="Evaluating checkpoints"):
        try:
            # Extract checkpoint number for naming
            checkpoint_num = "unknown"
            if "-checkpoint-" in checkpoint_model:
                checkpoint_num = checkpoint_model.split("-checkpoint-")[1]
            
            # Create evaluation config
            eval_config = Config.from_dict(config.to_dict())
            if eval_config.logging.experiment_name:
                eval_config.logging.experiment_name = f"{eval_config.logging.experiment_name}_checkpoint-{checkpoint_num}"
            else:
                eval_config.logging.experiment_name = f"checkpoint-{checkpoint_num}_evaluation"
            
            # Determine which dataset generator to use
            dataset_generator = None
            if args.skip_dataset_generation:
                # Use existing datasets with the appropriate file base
                from data.dataset import DatasetGenerator
                dataset_generator = DatasetGenerator(eval_config)
                dataset_generator.file_base = f"checkpoint-{checkpoint_num}"
                
                # Check if the dataset exists
                if not dataset_generator.dataset_exists():
                    print(f"Warning: Dataset for checkpoint-{checkpoint_num} not found. Skipping evaluation.")
                    continue
            elif args.shared_dataset:
                # Use the shared dataset
                dataset_generator = shared_dataset_generator
            else:
                # Use the dataset generated for this specific checkpoint
                from data.dataset import DatasetGenerator
                dataset_generator = DatasetGenerator(eval_config)
                dataset_generator.file_base = f"checkpoint-{checkpoint_num}"
            
            # Evaluate the checkpoint
            result = evaluate_checkpoint(checkpoint_model, eval_config, dataset_generator)
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_model}: {e}")
            evaluation_results.append({
                "checkpoint_model": checkpoint_model,
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
    run_evaluation_pipeline() 