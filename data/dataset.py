# data/dataset.py
import os
import json
import torch
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


class DatasetGenerator:
    """Class for generating and managing datasets for IRL experiments."""
    
    def __init__(self, config):
        """
        Initialize the dataset generator.
        
        Args:
            config: Configuration object with dataset settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create file base for saving
        self.file_base = (f"{config.model.model_pair}_n{config.dataset.num_samples}_"
                         f"t{config.dataset.max_new_tokens}_temp{config.dataset.temperature}")
        
        # Set random seed
        torch.manual_seed(config.dataset.seed)
        np.random.seed(config.dataset.seed)
        
        # Initialize paths
        self.original_output_path = os.path.join(config.dataset.cache_dir, f"{self.file_base}_original.json")
        self.detoxified_output_path = os.path.join(config.dataset.cache_dir, f"{self.file_base}_detoxified.json")
        
        # Initialize data containers
        self.prompts = []
        self.original_data = []
        self.detoxified_data = []
    
    def load_prompts(self):
        """Load toxic prompts from the RealToxicityPrompts dataset."""
        print("Loading RealToxicityPrompts dataset...")
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")

        # Filter for prompts with toxicity > 0.3 (same as in the original RLHF training)
        def filter_fn(sample):
            toxicity = sample["prompt"]["toxicity"]
            return toxicity is not None and toxicity > 0.3

        ds = ds.filter(filter_fn, batched=False)

        # Select the required number of samples
        ds = ds.select(range(min(self.config.dataset.num_samples, len(ds))))

        # Extract prompts
        self.prompts = [example["prompt"]["text"] for example in ds]
        
        print(f"Loaded {len(self.prompts)} prompts")
        return self.prompts
    
    def generate_completions(self, model_type="original"):
        """
        Generate completions from either the original or detoxified model.
        
        Args:
            model_type: "original" or "detoxified"
        
        Returns:
            List of completions
        """
        # Get model paths
        original_model_path, detoxified_model_path = self.config.model.get_model_paths()
        
        # Select the appropriate model
        if model_type == "original":
            model_path = original_model_path
        else:
            model_path = detoxified_model_path
        
        print(f"Loading {model_type} model: {model_path}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.model.use_half_precision else None
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Fix for decoder-only models

        print(f"Generating completions from the {model_type} model...")
        completions = []

        for i in tqdm(range(0, len(self.prompts), self.config.dataset.batch_size)):
            batch_prompts = self.prompts[i:i+self.config.dataset.batch_size]

            # Tokenize the batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate completions with specified parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.dataset.max_new_tokens,
                    do_sample=(self.config.dataset.temperature > 0),
                    temperature=self.config.dataset.temperature,
                    top_p=self.config.dataset.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode the outputs
            batch_completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract only the new tokens (remove the prompt)
            for j, completion in enumerate(batch_completions):
                # Replace the prompt to get only the generated part
                new_text = completion.replace(batch_prompts[j], "", 1).strip()
                completions.append(new_text)

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return completions
    
    def generate_datasets(self):
        """Generate paired datasets from both models."""
        # Check if we should use cached datasets
        if self.config.dataset.use_cached and self._check_cached_datasets():
            print("Using cached datasets")
            return self.load_datasets()
        
        # Load prompts if not already loaded
        if not self.prompts:
            self.load_prompts()
        
        # Generate completions from both models
        original_completions = self.generate_completions("original")
        detoxified_completions = self.generate_completions("detoxified")
        
        # Create paired datasets
        print("Creating paired datasets...")
        self.original_data = []
        self.detoxified_data = []

        for i in range(len(self.prompts)):
            self.original_data.append({
                "prompt": self.prompts[i],
                "output": original_completions[i],
                "model_type": "original"
            })

            self.detoxified_data.append({
                "prompt": self.prompts[i],
                "output": detoxified_completions[i],
                "model_type": "detoxified"
            })
        
        # Save datasets
        self._save_datasets()
        
        return self.original_data, self.detoxified_data
    
    def _save_datasets(self):
        """Save the generated datasets to disk."""
        print(f"Saving datasets...")
        
        # Ensure directories exist
        os.makedirs(self.config.dataset.cache_dir, exist_ok=True)
        
        # Save as JSON
        with open(self.original_output_path, "w") as f:
            json.dump(self.original_data, f, indent=2)

        with open(self.detoxified_output_path, "w") as f:
            json.dump(self.detoxified_data, f, indent=2)

        # Save as CSV for easier inspection
        pd.DataFrame(self.original_data).to_csv(
            os.path.join(self.config.dataset.cache_dir, f"{self.file_base}_original.csv"), index=False
        )
        pd.DataFrame(self.detoxified_data).to_csv(
            os.path.join(self.config.dataset.cache_dir, f"{self.file_base}_detoxified.csv"), index=False
        )
        
        # Save to Google Drive if configured
        if self.config.logging.save_to_drive and os.path.exists('/content/drive'):
            drive_datasets_dir = os.path.join(self.config.logging.drive_path, "datasets")
            os.makedirs(drive_datasets_dir, exist_ok=True)
            
            # Copy files to Drive
            with open(os.path.join(drive_datasets_dir, f"{self.file_base}_original.json"), "w") as f:
                json.dump(self.original_data, f, indent=2)

            with open(os.path.join(drive_datasets_dir, f"{self.file_base}_detoxified.json"), "w") as f:
                json.dump(self.detoxified_data, f, indent=2)

            pd.DataFrame(self.original_data).to_csv(
                os.path.join(drive_datasets_dir, f"{self.file_base}_original.csv"), index=False
            )
            pd.DataFrame(self.detoxified_data).to_csv(
                os.path.join(drive_datasets_dir, f"{self.file_base}_detoxified.csv"), index=False
            )
            
            print(f"Datasets also saved to Google Drive at {drive_datasets_dir}")
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # Log dataset metadata
            wandb.log({
                "dataset": {
                    "original_samples": len(self.original_data),
                    "detoxified_samples": len(self.detoxified_data),
                    "file_base": self.file_base
                }
            })
            
            # Log sample datasets to wandb
            original_table = wandb.Table(dataframe=pd.DataFrame(self.original_data[:10]))
            detoxified_table = wandb.Table(dataframe=pd.DataFrame(self.detoxified_data[:10]))
            
            wandb.log({
                "original_samples": original_table,
                "detoxified_samples": detoxified_table
            })
            
            # Save the full datasets as artifacts
            original_artifact = wandb.Artifact(f"original_dataset_{self.file_base}", type="dataset")
            original_artifact.add_file(self.original_output_path)
            wandb.log_artifact(original_artifact)
            
            detoxified_artifact = wandb.Artifact(f"detoxified_dataset_{self.file_base}", type="dataset")
            detoxified_artifact.add_file(self.detoxified_output_path)
            wandb.log_artifact(detoxified_artifact)
    
    def _check_cached_datasets(self):
        """Check if cached datasets exist."""
        return os.path.exists(self.original_output_path) and os.path.exists(self.detoxified_output_path)
    
    def load_datasets(self):
        """Load datasets from disk."""
        if not self._check_cached_datasets():
            print("Cached datasets not found, generating new ones...")
            return self.generate_datasets()
        
        print(f"Loading datasets from {self.config.dataset.cache_dir}...")
        
        with open(self.original_output_path, "r") as f:
            self.original_data = json.load(f)
            
        with open(self.detoxified_output_path, "r") as f:
            self.detoxified_data = json.load(f)
            
        # Extract prompts for reference
        self.prompts = [item["prompt"] for item in self.original_data]
        
        print(f"Loaded {len(self.original_data)} samples from each dataset")
        
        return self.original_data, self.detoxified_data
    
    def analyze_datasets(self):
        """Analyze and compare the generated datasets."""
        if not self.original_data or not self.detoxified_data:
            print("No datasets loaded. Loading from disk...")
            self.load_datasets()
            
        print("\n--- Dataset Analysis ---")
        print(f"Total samples: {len(self.original_data)}")

        # Sample inspection
        print("\nSample Inspection:")
        for i in range(min(5, len(self.original_data))):
            print(f"\nSample {i+1}:")
            print(f"Prompt: {self.original_data[i]['prompt'][:100]}...")
            print(f"Original output: {self.original_data[i]['output']}")
            print(f"Detoxified output: {self.detoxified_data[i]['output']}")

            # Check if outputs are different
            if self.original_data[i]['output'] == self.detoxified_data[i]['output']:
                print("⚠️ WARNING: Outputs are identical")
            else:
                print("✓ Outputs are different")

        # Check for empty generations
        original_empty = sum(1 for item in self.original_data if not item['output'])
        detoxified_empty = sum(1 for item in self.detoxified_data if not item['output'])

        print(f"\nEmpty generations in original model: {original_empty} "
              f"({original_empty/len(self.original_data)*100:.2f}%)")
        print(f"Empty generations in detoxified model: {detoxified_empty} "
              f"({detoxified_empty/len(self.detoxified_data)*100:.2f}%)")

        # Length statistics
        original_lengths = [len(item['output']) for item in self.original_data]
        detoxified_lengths = [len(item['output']) for item in self.detoxified_data]

        print(f"\nOriginal output length: avg={np.mean(original_lengths):.2f}, "
              f"min={min(original_lengths)}, max={max(original_lengths)}")
        print(f"Detoxified output length: avg={np.mean(detoxified_lengths):.2f}, "
              f"min={min(detoxified_lengths)}, max={max(detoxified_lengths)}")

        # Check for identical outputs
        identical_count = sum(1 for i in range(len(self.original_data)) 
                              if self.original_data[i]['output'] == self.detoxified_data[i]['output'])
        identical_pct = identical_count/len(self.original_data)*100
        print(f"\nIdentical outputs: {identical_count} ({identical_pct:.2f}%)")

        # Create analysis report
        report = {
            "total_samples": len(self.original_data),
            "original_empty": original_empty,
            "original_empty_pct": float(original_empty/len(self.original_data)*100),
            "detoxified_empty": detoxified_empty,
            "detoxified_empty_pct": float(detoxified_empty/len(self.detoxified_data)*100),
            "original_length_avg": float(np.mean(original_lengths)),
            "original_length_min": min(original_lengths),
            "original_length_max": max(original_lengths),
            "detoxified_length_avg": float(np.mean(detoxified_lengths)),
            "detoxified_length_min": min(detoxified_lengths),
            "detoxified_length_max": max(detoxified_lengths),
            "identical_outputs": identical_count,
            "identical_pct": float(identical_pct)
        }
        
        # Save the analysis
        analysis_path = os.path.join(self.config.dataset.cache_dir, f"{self.file_base}_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Save to Google Drive if configured
        if self.config.logging.save_to_drive and os.path.exists('/content/drive'):
            drive_analysis_path = os.path.join(self.config.logging.drive_path, "datasets", 
                                              f"{self.file_base}_analysis.json")
            with open(drive_analysis_path, "w") as f:
                json.dump(report, f, indent=2)
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            wandb.log({
                "dataset_analysis": report,
                "original_lengths_hist": wandb.Histogram(original_lengths),
                "detoxified_lengths_hist": wandb.Histogram(detoxified_lengths)
            })
        
        print("\nAnalysis complete!")
        return report


def load_or_generate_dataset(config):
    """
    Helper function to load or generate datasets based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        original_data, detoxified_data, file_base
    """
    generator = DatasetGenerator(config)
    
    if config.dataset.use_cached and generator._check_cached_datasets():
        original_data, detoxified_data = generator.load_datasets()
    else:
        original_data, detoxified_data = generator.generate_datasets()
    
    # Analyze the datasets
    generator.analyze_datasets()
    
    return original_data, detoxified_data, generator.file_base