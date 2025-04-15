# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import wandb


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_training_metrics(metrics_history, save_path=None, show=True, log_wandb=False):
    """
    Plot training metrics history.
    
    Args:
        metrics_history: List of metrics dictionaries
        save_path: Optional path to save the plot
        show: Whether to show the plot
        log_wandb: Whether to log the plot to wandb
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    # Extract epochs and metrics
    epochs = [m['epoch'] for m in metrics_history]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot accuracy and F1
    ax1 = axes[0, 0]
    ax1.plot(epochs, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
    ax1.plot(epochs, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    ax1.plot(epochs, [m['auc_roc'] for m in metrics_history], '^-', label='AUC-ROC')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot correlations
    ax2 = axes[0, 1]
    ax2.plot(epochs, [m['pearson_correlation'] for m in metrics_history], 'o-', label='Pearson')
    ax2.plot(epochs, [m['spearman_correlation'] for m in metrics_history], 's-', label='Spearman')
    ax2.plot(epochs, [m['kendall_tau'] for m in metrics_history], '^-', label='Kendall Tau')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation with True Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot average rewards
    ax3 = axes[1, 0]
    ax3.plot(epochs, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
    ax3.plot(epochs, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
    ax3.plot(epochs, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Average Predicted Rewards')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot loss
    ax4 = axes[1, 1]
    ax4.plot(epochs, [m['loss'] for m in metrics_history], 'o-')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if log_wandb and wandb.run is not None:
        wandb.log({"training_metrics": wandb.Image(fig)})
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_score_distribution(original_scores, detoxified_scores, model_name="model", 
                           save_path=None, show=True, log_wandb=False):
    """
    Plot distribution of scores for toxic vs non-toxic content.
    
    Args:
        original_scores: Scores for original (toxic) content
        detoxified_scores: Scores for detoxified content
        model_name: Name of the model
        save_path: Optional path to save the plot
        show: Whether to show the plot
        log_wandb: Whether to log the plot to wandb
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    ax.hist(original_scores, alpha=0.5, bins=20, label='Original (Toxic)', color='red')
    ax.hist(detoxified_scores, alpha=0.5, bins=20, label='Detoxified', color='green')
    
    # Plot means as vertical lines
    ax.axvline(np.mean(original_scores), color='red', linestyle='--', 
              label=f'Mean Original: {np.mean(original_scores):.4f}')
    ax.axvline(np.mean(detoxified_scores), color='green', linestyle='--',
              label=f'Mean Detoxified: {np.mean(detoxified_scores):.4f}')
    
    # Add labels and title
    ax.set_xlabel('Reward Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Reward Scores - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with summary statistics
    diff = np.mean(detoxified_scores) - np.mean(original_scores)
    text = (f"Mean Difference: {diff:.4f}\n"
           f"Original Std: {np.std(original_scores):.4f}\n"
           f"Detoxified Std: {np.std(detoxified_scores):.4f}")
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if log_wandb and wandb.run is not None:
        wandb.log({
            "score_distribution": wandb.Image(fig),
            "mean_original_score": np.mean(original_scores),
            "mean_detoxified_score": np.mean(detoxified_scores),
            "score_difference": diff,
            "original_scores_std": np.std(original_scores),
            "detoxified_scores_std": np.std(detoxified_scores)
        })
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_learning_curve(metrics_history, metric_name, title=None, save_path=None, show=True, log_wandb=False):
    """
    Plot learning curve for a specific metric.
    
    Args:
        metrics_history: List of metrics dictionaries
        metric_name: Name of the metric to plot
        title: Optional title for the plot
        save_path: Optional path to save the plot
        show: Whether to show the plot
        log_wandb: Whether to log the plot to wandb
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    # Extract epochs and metric
    epochs = [m['epoch'] for m in metrics_history]
    metric_values = [m[metric_name] for m in metrics_history]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metric
    ax.plot(epochs, metric_values, 'o-')
    
    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Learning Curve - {metric_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if log_wandb and wandb.run is not None:
        wandb.log({f"learning_curve_{metric_name}": wandb.Image(fig)})
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_dataset_analysis(dataset_analysis, save_path=None, show=True, log_wandb=False):
    """
    Plot dataset analysis results.
    
    Args:
        dataset_analysis: Dictionary with dataset analysis results
        save_path: Optional path to save the plot
        show: Whether to show the plot
        log_wandb: Whether to log the plot to wandb
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    # Create figure with 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot length distributions
    ax1 = axes[0]
    data = {
        'Original': {
            'avg': dataset_analysis['original_length_avg'],
            'min': dataset_analysis['original_length_min'],
            'max': dataset_analysis['original_length_max']
        },
        'Detoxified': {
            'avg': dataset_analysis['detoxified_length_avg'],
            'min': dataset_analysis['detoxified_length_min'],
            'max': dataset_analysis['detoxified_length_max']
        }
    }
    df = pd.DataFrame(data).T
    df.plot(kind='bar', ax=ax1)
    ax1.set_title('Output Length Statistics')
    ax1.set_ylabel('Character Count')
    ax1.set_ylim(bottom=0)
    
    # Add text labels above bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f')
    
    # Plot empty and identical percentages
    ax2 = axes[1]
    data = {
        'Original Empty': dataset_analysis['original_empty_pct'],
        'Detoxified Empty': dataset_analysis['detoxified_empty_pct'],
        'Identical Outputs': dataset_analysis['identical_pct']
    }
    colors = ['#FF9999', '#99FF99', '#9999FF']
    ax2.bar(data.keys(), data.values(), color=colors)
    ax2.set_title('Dataset Quality Metrics')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, max(data.values()) * 1.2)  # Add some headroom
    
    # Add text labels above bars
    for i, (key, value) in enumerate(data.items()):
        ax2.text(i, value + 1, f'{value:.1f}%', ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if log_wandb and wandb.run is not None:
        wandb.log({"dataset_analysis": wandb.Image(fig)})
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig