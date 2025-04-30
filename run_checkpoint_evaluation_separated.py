import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Run checkpoint evaluation pipeline")
    parser.add_argument("--generate_only", action="store_true", 
                        help="Only generate datasets, don't evaluate")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only evaluate existing datasets")
    parser.add_argument("--dataset_results", type=str, default=None,
                        help="Path to dataset generation results JSON file (for evaluate_only)")
    
    # Add all the arguments from generate_fresh_datasets.py
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, 
                        help="Base model to compare against (e.g., 'EleutherAI/pythia-70m')")
    parser.add_argument("--checkpoint_repo", type=str,
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
    parser.add_argument("--external_prompts", type=str, default=None,
                        help="Path to a JSON file with external prompts")
    parser.add_argument("--external_dataset", type=str, default=None,
                        help="Name of HuggingFace dataset to use for prompts")
    
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

def create_args_dict(args):
    """Convert args to a dictionary, excluding None values"""
    return {k: v for k, v in vars(args).items() if v is not None}

if __name__ == "__main__":
    args = parse_args()
    args_dict = create_args_dict(args)
    
    if args.evaluate_only:
        # Run evaluation only
        from scripts.evaluate_fresh_datasets import main as evaluate_datasets
        
        # Remove generate-only arguments
        for key in ['generate_only', 'evaluate_only']:
            if key in args_dict:
                del args_dict[key]
        
        # Set up the configuration for evaluate_datasets
        from config.fresh_config import FreshConfig
        config = FreshConfig()
        
        # Update config with args
        for key, value in args_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.logging, key):
                setattr(config.logging, key, value)
        
        # Run evaluation
        evaluate_datasets(config=config)
        
    elif args.generate_only:
        # Run generation only
        from scripts.generate_fresh_datasets import main as generate_datasets
        
        # Remove evaluate-only arguments
        for key in ['generate_only', 'evaluate_only', 'dataset_results']:
            if key in args_dict:
                del args_dict[key]
        
        # Check required arguments
        if 'base_model' not in args_dict or 'checkpoint_repo' not in args_dict:
            print("Error: --base_model and --checkpoint_repo are required for dataset generation")
            sys.exit(1)
        
        # Set up the configuration for generate_datasets
        from config.fresh_config import FreshConfig
        config = FreshConfig()
        
        # Update config with args
        for key, value in args_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.logging, key):
                setattr(config.logging, key, value)
        
        # Run generation
        generate_datasets(config=config)
        
    else:
        # Run both
        from scripts.generate_fresh_datasets import main as generate_datasets
        from scripts.evaluate_fresh_datasets import main as evaluate_datasets
        
        # Check required arguments
        if 'base_model' not in args_dict or 'checkpoint_repo' not in args_dict:
            print("Error: --base_model and --checkpoint_repo are required for dataset generation")
            sys.exit(1)
        
        # Set up the configuration
        from config.fresh_config import FreshConfig
        config = FreshConfig()
        
        # Update config with args
        for key, value in args_dict.items():
            if key in ['generate_only', 'evaluate_only', 'dataset_results']:
                continue
                
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.logging, key):
                setattr(config.logging, key, value)
        
        # Generate datasets
        generate_datasets(config=config)
        
        # Evaluate datasets
        evaluate_datasets(config=config) 