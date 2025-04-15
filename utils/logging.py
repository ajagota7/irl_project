# utils/logging.py
import os
import json
import wandb
from datetime import datetime


def setup_wandb(config, run_name=None):
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        config: Configuration object
        run_name: Optional run name
        
    Returns:
        wandb run object or None if wandb is disabled
    """
    if not config.logging.use_wandb:
        print("WandB logging is disabled")
        return None
    
    # Create experiment name if not provided
    if not run_name and not config.logging.experiment_name:
        run_name = f"{config.model.model_pair}_{config.dataset.num_samples}samples_{config.training.epochs}epochs"
    elif not run_name:
        run_name = config.logging.experiment_name
    
    # Initialize wandb
    run = wandb.init(
        project=config.logging.project_name,
        name=run_name,
        config=config.to_dict(),
        mode=config.logging.wandb_mode
    )
    
    print(f"WandB initialized with run name: {run_name}")
    return run


def save_config(config, path):
    """
    Save configuration to a file.
    
    Args:
        config: Configuration object
        path: Path to save the configuration
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save config
    config.save(path)
    print(f"Configuration saved to {path}")


def log_artifact(name, artifact_type, data_or_path, metadata=None):
    """
    Log an artifact to WandB.
    
    Args:
        name: Artifact name
        artifact_type: Artifact type
        data_or_path: Data or path to save
        metadata: Optional metadata
    """
    if wandb.run is None:
        print("WandB run not initialized, can't log artifact")
        return
    
    # Create artifact
    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
    
    # Add data or file to artifact
    if isinstance(data_or_path, str) and os.path.exists(data_or_path):
        if os.path.isdir(data_or_path):
            artifact.add_dir(data_or_path)
        else:
            artifact.add_file(data_or_path)
    elif isinstance(data_or_path, dict):
        # Save dict as JSON file
        with open(f"{name}.json", "w") as f:
            json.dump(data_or_path, f, indent=2)
        artifact.add_file(f"{name}.json")
    
    # Log artifact
    wandb.log_artifact(artifact)
    print(f"Logged artifact {name} to WandB")


def log_metrics(metrics, step=None):
    """
    Log metrics to WandB.
    
    Args:
        metrics: Dictionary of metrics
        step: Optional step number
    """
    if wandb.run is None:
        print("WandB run not initialized, can't log metrics")
        return
    
    # Log metrics
    wandb.log(metrics, step=step)


class ExperimentLogger:
    """Logger class for experiment tracking."""
    
    def __init__(self, config, experiment_name=None):
        """
        Initialize the logger.
        
        Args:
            config: Configuration object
            experiment_name: Optional experiment name
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment name if not provided
        if not experiment_name:
            self.experiment_name = f"{config.model.model_pair}_{self.timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Set up directories
        self.log_dir = os.path.join(config.logging.log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up WandB if configured
        if config.logging.use_wandb:
            self.wandb_run = setup_wandb(config, run_name=self.experiment_name)
        else:
            self.wandb_run = None
        
        # Save config
        self.config_path = os.path.join(self.log_dir, "config.yaml")
        save_config(config, self.config_path)
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        # Log to wandb if configured
        if self.wandb_run is not None:
            log_metrics(metrics, step)
        
        # Save metrics to file
        metrics_path = os.path.join(self.log_dir, f"metrics_{step if step is not None else 'latest'}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def log_artifact(self, name, artifact_type, data_or_path, metadata=None):
        """
        Log an artifact.
        
        Args:
            name: Artifact name
            artifact_type: Artifact type
            data_or_path: Data or path to save
            metadata: Optional metadata
        """
        # Log to wandb if configured
        if self.wandb_run is not None:
            log_artifact(name, artifact_type, data_or_path, metadata)
        
        # Save artifact to disk if it's a dictionary
        if isinstance(data_or_path, dict):
            artifact_path = os.path.join(self.log_dir, f"{name}.json")
            with open(artifact_path, "w") as f:
                json.dump(data_or_path, f, indent=2)
    
    def log_model(self, model, model_name="model"):
        """
        Log a model.
        
        Args:
            model: Model to save
            model_name: Name of the model
        """
        # Save model to disk
        model_dir = os.path.join(self.log_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, "model.pt"))
        
        # Log to wandb if configured
        if self.wandb_run is not None:
            artifact = wandb.Artifact(model_name, type="model")
            artifact.add_dir(model_dir)
            self.wandb_run.log_artifact(artifact)
    
    def finish(self):
        """Finish logging."""
        if self.wandb_run is not None:
            self.wandb_run.finish()