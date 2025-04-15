# Modular IRL Framework for Detoxification

This repository contains a modular framework for Inverse Reinforcement Learning (IRL) focused on learning reward models from pairs of toxic and detoxified text generations. It provides tools for dataset generation, model training, evaluation, and experiment tracking with Weights & Biases (WandB).

## Project Structure

```
irl_project/
├── config/             # Configuration classes and utilities
├── data/               # Dataset generation and loading functions
├── models/             # Reward model definition
├── training/           # Training and evaluation utilities
├── utils/              # Visualization and logging utilities
├── scripts/            # Executable scripts
├── main.py             # Main entry point
└── requirements.txt    # Project dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/irl-detoxification.git
cd irl-detoxification

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up WandB for experiment tracking
wandb login
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each aspect of the IRL pipeline.
- **Experiment Tracking**: Integrated WandB support for tracking experiments, metrics, and artifacts.
- **Configurable**: Comprehensive configuration system with YAML support and command-line overrides.
- **Google Colab Support**: Optional integration with Google Drive for saving/loading datasets and models in Colab.
- **Comprehensive Evaluation**: Rich evaluation metrics and visualizations.

## Usage

### Dataset Generation

Generate paired datasets from original and detoxified language models:

```bash
python scripts/generate_dataset.py --model_pair gpt-neo-125M --num_samples 1000 --use_wandb
```

### Model Training

Train a reward model using the generated datasets:

```bash
python scripts/train_model.py --model_pair gpt-neo-125M --epochs 30 --batch_size 8 --use_wandb
```

### Model Evaluation

Evaluate a trained reward model:

```bash
python scripts/evaluate_model.py --model_path ./models/gpt-neo-125M_date/model.pt --use_wandb
```

### Running the Full Pipeline

Run the entire pipeline from dataset generation to evaluation:

```bash
python main.py --mode all --model_pair gpt-neo-125M --num_samples 1000 --epochs 30 --use_wandb
```

## Configuration

You can customize the behavior using YAML configuration files:

```bash
python main.py --config config/sample_config.yaml
```

See `config/sample_config.yaml` for a complete example.

## Customization

### Adding a New Model

1. Update the model paths in `config/config.py`:

```python
def get_model_paths(model_pair):
    if model_pair == "your-new-model":
        original_model = "path/to/original_model"
        detoxified_model = "path/to/detoxified_model"
    # ...
```

### Adding New Evaluation Metrics

Add new metrics in `training/evaluation.py` and update the `evaluate` method in `RewardModelTrainer`.

## WandB Integration

This framework integrates with Weights & Biases for experiment tracking:

- Logs hyperparameters and configurations
- Tracks metrics during training and evaluation
- Saves model checkpoints as artifacts
- Visualizes datasets and evaluation results

Enable WandB logging with the `--use_wandb` flag or in the configuration file.

## Google Drive Integration (for Colab)

When running on Google Colab, you can save results to Google Drive:

```bash
python main.py --save_to_drive --drive_path "/content/drive/MyDrive/irl_experiments"
```
