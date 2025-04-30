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
    base_model: str = "EleutherAI/pythia-70m"  # Base model to compare against
    checkpoint_repo: str = "ajagota71/pythia-70m-detox"  # Repository with checkpoints
    checkpoint_pattern: str = "checkpoint-epoch-"  # Pattern to match checkpoint folders
    specific_checkpoints: Optional[List[str]] = None  # Specific checkpoints to evaluate
    max_checkpoints: Optional[int] = None  # Maximum number of checkpoints to evaluate
    checkpoint_step: int = 1  # Evaluate every Nth checkpoint
    use_half_precision: bool = False  # Whether to use half precision
    
    def __post_init__(self):
        # Automatically determine if we should use half precision based on model size
        if "2.7B" in self.base_model or "6B" in self.base_model or "7B" in self.base_model:
            self.use_half_precision = True


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_samples: int = 100
    max_new_tokens: int = 30
    batch_size: int = 16
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    use_cached: bool = False
    cache_dir: str = "./datasets"
    shared_dataset: bool = True  # Whether to use the same prompts for all checkpoints
    skip_dataset_generation: bool = False  # Skip dataset generation and use existing datasets


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
    unfrozen_layers: int = 1  # Number of layers to unfreeze for training


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    project_name: str = "checkpoint-evaluation"
    experiment_name: Optional[str] = None
    log_dir: str = "./logs"
    use_wandb: bool = False
    wandb_mode: str = "online"  # online, offline, disabled
    eval_dir: str = "./evaluation"
    save_model: bool = True
    model_dir: str = "./models"
    save_to_drive: bool = False
    drive_path: str = "/content/drive/MyDrive/checkpoint_evaluation"


@dataclass
class FreshConfig:
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FreshConfig':
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
    def from_yaml(cls, path: str) -> 'FreshConfig':
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
            run_name = f"{self.model.base_model.split('/')[-1]}_{self.dataset.num_samples}samples"
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
def get_default_config() -> FreshConfig:
    """Get default configuration."""
    return FreshConfig()


def get_config_from_args(args) -> FreshConfig:
    """Create configuration from command-line arguments."""
    config = get_default_config()
    
    # Update model config
    if hasattr(args, 'base_model'):
        config.model.base_model = args.base_model
    if hasattr(args, 'checkpoint_repo'):
        config.model.checkpoint_repo = args.checkpoint_repo
    if hasattr(args, 'checkpoint_pattern'):
        config.model.checkpoint_pattern = args.checkpoint_pattern
    if hasattr(args, 'specific_checkpoints') and args.specific_checkpoints:
        config.model.specific_checkpoints = args.specific_checkpoints.split(',')
    if hasattr(args, 'max_checkpoints'):
        config.model.max_checkpoints = args.max_checkpoints
    if hasattr(args, 'checkpoint_step'):
        config.model.checkpoint_step = args.checkpoint_step
    
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
    if hasattr(args, 'cache_dir'):
        config.dataset.cache_dir = args.cache_dir
    if hasattr(args, 'shared_dataset'):
        config.dataset.shared_dataset = args.shared_dataset
    if hasattr(args, 'skip_dataset_generation'):
        config.dataset.skip_dataset_generation = args.skip_dataset_generation
    
    # Update training config
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    if hasattr(args, 'train_batch_size'):
        config.training.batch_size = args.train_batch_size
    if hasattr(args, 'eval_interval'):
        config.training.eval_interval = args.eval_interval
    if hasattr(args, 'unfrozen_layers'):
        config.training.unfrozen_layers = args.unfrozen_layers
        
    # Update logging config
    if hasattr(args, 'project_name'):
        config.logging.project_name = args.project_name
    if hasattr(args, 'experiment_name'):
        config.logging.experiment_name = args.experiment_name
    if hasattr(args, 'use_wandb'):
        config.logging.use_wandb = args.use_wandb
    if hasattr(args, 'save_to_drive'):
        config.logging.save_to_drive = args.save_to_drive
    if hasattr(args, 'eval_dir'):
        config.logging.eval_dir = args.eval_dir
    
    return config 