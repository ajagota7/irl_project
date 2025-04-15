# config/config.py
import os
import torch
import dataclasses
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import yaml
import wandb


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_pair: str = "gpt-neo-125M"  # Choose from "gpt-neo-125M", "gpt-neo-2.7B", "gpt-j-6B"
    use_half_precision: bool = None  # Auto-determined based on model size
    
    def __post_init__(self):
        # Automatically determine if we should use half precision
        if self.use_half_precision is None:
            self.use_half_precision = "2.7B" in self.model_pair or "6B" in self.model_pair
    
    def get_model_paths(self):
        """Get the paths to the original and detoxified models."""
        if self.model_pair == "gpt-neo-125M":
            original_model = "EleutherAI/gpt-neo-125M"
            detoxified_model = "ybelkada/gpt-neo-125m-detox"
        elif self.model_pair == "gpt-neo-2.7B":
            original_model = "EleutherAI/gpt-neo-2.7B"
            detoxified_model = "ybelkada/gpt-neo-2.7B-detox"
        elif self.model_pair == "gpt-j-6B":
            original_model = "EleutherAI/gpt-j-6B"
            detoxified_model = "ybelkada/gpt-j-6b-detox"
        else:
            raise ValueError(f"Unknown model pair: {self.model_pair}")

        return original_model, detoxified_model


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_samples: int = 100
    max_new_tokens: int = 30
    batch_size: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    use_cached: bool = False
    cache_dir: str = "./datasets"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-5
    epochs: int = 20
    batch_size: int = 4
    eval_interval: int = 5
    max_length: int = 512
    train_test_split: float = 0.8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    margin: float = 0.1
    adam_epsilon: float = 1e-8
    seed: int = 42


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    project_name: str = "irl-experiments"
    experiment_name: Optional[str] = None
    log_dir: str = "./logs"
    use_wandb: bool = True
    wandb_mode: str = "online"  # online, offline, disabled
    eval_dir: str = "./evaluation"
    save_model: bool = True
    model_dir: str = "./models"
    save_to_drive: bool = False
    drive_path: str = "/content/drive/MyDrive/irl_experiments"


@dataclass
class Config:
    """Master configuration class."""
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return dataclasses.asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            logging=logging_config
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def init_wandb(self, run_name: Optional[str] = None) -> None:
        """Initialize Weights & Biases."""
        if not self.logging.use_wandb:
            return None
        
        # Create experiment name if not provided
        if not run_name and not self.logging.experiment_name:
            run_name = f"{self.model.model_pair}_{self.dataset.num_samples}samples_{self.training.epochs}epochs"
        elif not run_name:
            run_name = self.logging.experiment_name
        
        # Initialize wandb
        return wandb.init(
            project=self.logging.project_name,
            name=run_name,
            config=self.to_dict(),
            mode=self.logging.wandb_mode
        )
    
    def setup_directories(self) -> None:
        """Set up necessary directories."""
        os.makedirs(self.dataset.cache_dir, exist_ok=True)
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.eval_dir, exist_ok=True)
        os.makedirs(self.logging.model_dir, exist_ok=True)
        
        # Set up Google Drive directories if needed
        if self.logging.save_to_drive and os.path.exists('/content/drive'):
            drive_path = self.logging.drive_path
            os.makedirs(drive_path, exist_ok=True)
            os.makedirs(os.path.join(drive_path, "datasets"), exist_ok=True)
            os.makedirs(os.path.join(drive_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(drive_path, "evaluation"), exist_ok=True)
            os.makedirs(os.path.join(drive_path, "logs"), exist_ok=True)


# Example usage
def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_config_from_args(args) -> Config:
    """Create configuration from command-line arguments."""
    config = get_default_config()
    
    # Update model config
    if hasattr(args, 'model_pair'):
        config.model.model_pair = args.model_pair
    
    # Update dataset config
    if hasattr(args, 'num_samples'):
        config.dataset.num_samples = args.num_samples
    if hasattr(args, 'max_new_tokens'):
        config.dataset.max_new_tokens = args.max_new_tokens
    if hasattr(args, 'batch_size') and not hasattr(args, 'train_batch_size'):
        config.dataset.batch_size = args.batch_size
    if hasattr(args, 'temperature'):
        config.dataset.temperature = args.temperature
    if hasattr(args, 'top_p'):
        config.dataset.top_p = args.top_p
    if hasattr(args, 'seed'):
        config.dataset.seed = args.seed
        config.training.seed = args.seed
    if hasattr(args, 'use_cached'):
        config.dataset.use_cached = args.use_cached
    
    # Update training config
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    if hasattr(args, 'train_batch_size'):
        config.training.batch_size = args.train_batch_size
    if hasattr(args, 'eval_interval'):
        config.training.eval_interval = args.eval_interval
        
    # Update logging config
    if hasattr(args, 'project_name'):
        config.logging.project_name = args.project_name
    if hasattr(args, 'experiment_name'):
        config.logging.experiment_name = args.experiment_name
    if hasattr(args, 'use_wandb'):
        config.logging.use_wandb = args.use_wandb
    if hasattr(args, 'save_to_drive'):
        config.logging.save_to_drive = args.save_to_drive
    
    return config