import argparse
import os
import torch
import wandb
from huggingface_hub import list_models
from tqdm import tqdm

from config.config import Config, get_config_from_args
from data.dataset import DatasetGenerator


def parse_args():
    """Parse command line arguments for dataset generation from checkpoints."""
    parser = argparse.ArgumentParser(description="Generate datasets from checkpoint models")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name on Hugging Face (e.g., 'username/model-name')")
    parser.add_argument("--checkpoint_pattern", type=str, default="checkpoint-",
                        help="Pattern to match checkpoint names")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                        help="Maximum number of checkpoints to use (None for all)")
    parser.add_argument("--checkpoint_step", type=int, default=1,
                        help="Use every Nth checkpoint")
    parser.add_argument("--specific_checkpoints", type=str, default=None,
                        help="Comma-separated list of specific checkpoint numbers to use")
    
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
    parser.add_argument("--output_dir", type=str, default="./datasets", 
                        help="Output directory for datasets")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="irl-checkpoint-datasets", 
                        help="Project name for wandb")
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


def generate_datasets_from_checkpoints():
    """Generate datasets from multiple checkpoint models."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_config_from_args(args)
    
    # Update config with command line arguments
    if args.output_dir:
        config.dataset.cache_dir = args.output_dir
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
    if args.project_name:
        config.logging.project_name = args.project_name
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
    
    # Generate datasets for each checkpoint
    results = []
    for checkpoint_model in tqdm(checkpoint_models, desc="Generating datasets"):
        try:
            # Extract checkpoint number for naming
            checkpoint_num = "unknown"
            if "-checkpoint-" in checkpoint_model:
                checkpoint_num = checkpoint_model.split("-checkpoint-")[1]
            
            # Create a unique experiment name
            experiment_name = f"dataset_checkpoint-{checkpoint_num}"
            
            # Initialize wandb for this checkpoint
            if config.logging.use_wandb:
                wandb_run = wandb.init(
                    project=config.logging.project_name,
                    name=experiment_name,
                    config=config.to_dict(),
                    mode=config.logging.wandb_mode,
                    reinit=True
                )
                wandb.config.update({"checkpoint_model": checkpoint_model})
            else:
                wandb_run = None
            
            # Initialize dataset generator
            generator = DatasetGenerator(config)
            
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
                dataset_artifact.add_dir(config.dataset.cache_dir)
                wandb.log_artifact(dataset_artifact)
                
                # Finish wandb run
                wandb.finish()
            
            # Record results
            results.append({
                "checkpoint_model": checkpoint_model,
                "checkpoint_num": checkpoint_num,
                "file_base": generator.file_base,
                "analysis": analysis
            })
            
            print(f"Dataset generation complete for {checkpoint_model}")
            
        except Exception as e:
            print(f"Error generating datasets for {checkpoint_model}: {e}")
            results.append({
                "checkpoint_model": checkpoint_model,
                "error": str(e)
            })
    
    # Save overall results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.dataset.cache_dir, f"dataset_generation_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Dataset generation complete. Results saved to {results_file}")


if __name__ == "__main__":
    generate_datasets_from_checkpoints() 