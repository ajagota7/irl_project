import os
import sys
import argparse
import torch
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import list_repo_files

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.fresh_config import FreshConfig, get_config_from_args
from data.fresh_dataset import FreshDatasetGenerator


def parse_args():
    """Parse command line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate datasets from checkpoint models")
    
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
    parser.add_argument("--external_prompts", type=str, default=None,
                        help="Path to a JSON file with external prompts")
    parser.add_argument("--external_dataset", type=str, default=None,
                        help="Name of HuggingFace dataset to use for prompts (e.g., 'allenai/real-toxicity-prompts')")
    
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
    
    # Filter by specific checkpoints if provided
    if specific_checkpoints:
        specific_nums = [str(num) for num in specific_checkpoints]
        checkpoint_folders = [folder for folder in checkpoint_folders 
                             if any(num in folder for num in specific_nums)]
    
    # Apply step and max_checkpoints
    checkpoint_folders = checkpoint_folders[::checkpoint_step]
    if max_checkpoints:
        checkpoint_folders = checkpoint_folders[:max_checkpoints]
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders: {checkpoint_folders}")
    
    return checkpoint_folders


def extract_checkpoint_num(checkpoint_folder, pattern="checkpoint-epoch-"):
    """Extract the checkpoint number from the folder name."""
    if pattern in checkpoint_folder:
        try:
            return int(checkpoint_folder.replace(pattern, ""))
        except ValueError:
            return checkpoint_folder
    return checkpoint_folder


def main():
    """Main function for dataset generation."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = get_config_from_args(args)
    
    # Create evaluation directory
    os.makedirs(config.dataset.cache_dir, exist_ok=True)
    
    # Get checkpoint folders
    checkpoint_folders = get_checkpoint_folders(
        config.model.checkpoint_repo,
        config.model.checkpoint_pattern,
        config.model.max_checkpoints,
        config.model.checkpoint_step,
        config.model.specific_checkpoints
    )
    
    if not checkpoint_folders:
        print("No checkpoint folders found. Exiting.")
        return
    
    # Generate datasets
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle external prompts
    external_prompts = None
    if args.external_prompts or args.external_dataset:
        print("\n=== Loading external prompts ===")
        shared_dataset_generator = FreshDatasetGenerator(config)
        shared_dataset_generator.file_base = "external"
        
        try:
            external_prompts = shared_dataset_generator.load_external_prompts(
                dataset_path=args.external_prompts,
                dataset_name=args.external_dataset if args.external_dataset else "allenai/real-toxicity-prompts",
                num_samples=config.dataset.num_samples
            )
            print(f"External prompts loaded successfully")
        except Exception as e:
            print(f"Error loading external prompts: {e}")
            external_prompts = None

    # Generate shared dataset if needed and external prompts not available
    if config.dataset.shared_dataset and not external_prompts:
        print("\n=== Generating shared dataset ===")
        shared_dataset_generator = FreshDatasetGenerator(config)
        shared_dataset_generator.file_base = "shared"
        
        # Generate datasets for the first checkpoint
        first_checkpoint = checkpoint_folders[0]
        first_checkpoint_num = extract_checkpoint_num(first_checkpoint)
        
        try:
            base_data, checkpoint_data = shared_dataset_generator.generate_datasets(
                config.model.base_model,
                config.model.checkpoint_repo,
                first_checkpoint
            )
            
            # Analyze datasets
            dataset_analysis = shared_dataset_generator.analyze_datasets()
            
            print(f"Shared dataset generation complete")
            
        except Exception as e:
            print(f"Error generating shared dataset: {e}")
            return
    
    # Generate datasets for each checkpoint
    for i, checkpoint_folder in enumerate(tqdm(checkpoint_folders, desc="Generating datasets")):
        checkpoint_num = extract_checkpoint_num(checkpoint_folder)
        
        print(f"\n{'='*80}")
        print(f"Generating dataset for checkpoint: {checkpoint_folder} (Epoch {checkpoint_num})")
        print(f"{'='*80}")
        
        try:
            if external_prompts:
                # Use external prompts
                print(f"Using external prompts for {checkpoint_folder}...")
                dataset_generator = FreshDatasetGenerator(config)
                dataset_generator.file_base = f"{checkpoint_num}"
                
                # Generate datasets for this checkpoint using external prompts
                base_data, checkpoint_data = dataset_generator.generate_datasets(
                    config.model.base_model,
                    config.model.checkpoint_repo,
                    checkpoint_folder,
                    prompts=external_prompts
                )
            elif shared_dataset_generator:
                # Use shared dataset generator
                print(f"Using shared dataset for {checkpoint_folder}...")
                dataset_generator = FreshDatasetGenerator(config)
                dataset_generator.file_base = f"{checkpoint_num}"
                
                # Generate datasets for this checkpoint using the same prompts
                base_data, checkpoint_data = dataset_generator.generate_datasets(
                    config.model.base_model,
                    config.model.checkpoint_repo,
                    checkpoint_folder,
                    prompts=shared_dataset_generator.load_prompts()
                )
            else:
                # Create a new dataset generator for this checkpoint
                dataset_generator = FreshDatasetGenerator(config)
                dataset_generator.file_base = f"{checkpoint_num}"
                
                # Generate datasets for this checkpoint
                base_data, checkpoint_data = dataset_generator.generate_datasets(
                    config.model.base_model,
                    config.model.checkpoint_repo,
                    checkpoint_folder
                )
            
            # Analyze datasets
            dataset_analysis = dataset_generator.analyze_datasets()
            
            # Record results
            results.append({
                "checkpoint_folder": checkpoint_folder,
                "checkpoint_num": checkpoint_num,
                "file_base": dataset_generator.file_base,
                "dataset_analysis": dataset_analysis
            })
            
            print(f"Dataset generation complete for {checkpoint_folder}")
            
        except Exception as e:
            print(f"Error generating datasets for {checkpoint_folder}: {e}")
            results.append({
                "checkpoint_folder": checkpoint_folder,
                "checkpoint_num": checkpoint_num,
                "error": str(e)
            })
    
    # Save overall results
    results_file = os.path.join(config.dataset.cache_dir, f"dataset_generation_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Dataset generation complete. Results saved to {results_file}")


if __name__ == "__main__":
    main() 