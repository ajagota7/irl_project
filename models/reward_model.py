# models/reward_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RewardModel(nn.Module):
    """Reward model that predicts whether text is toxic or not."""
    
    def __init__(self, model_name, use_half_precision=False, device="cuda", num_unfrozen_layers=1):
        """
        Initialize the reward model with a value head on top of a language model.
        
        Args:
            model_name: Name or path of the base language model
            use_half_precision: Whether to use half precision (float16)
            device: Device to use (cuda or cpu)
            num_unfrozen_layers: Number of layers to unfreeze for training (from the end)
        """
        super().__init__()
        
        # Set up device and precision
        self.device = device
        self.use_half_precision = use_half_precision
        
        # Load the base LM with the appropriate precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_half_precision else None,
            device_map="auto" if device == "cuda" else None
        )
        
        # Add a value head with careful initialization
        self.v_head = nn.Linear(self.model.config.hidden_size, 1, bias=False).to(device)
        # Initialize with small values to avoid NaN issues
        self.v_head.weight.data.normal_(mean=0.0, std=0.01)
            
        # Freeze the base model if it's large
        self._freeze_base_model(num_unfrozen_layers)
    
    def _freeze_base_model(self, num_unfrozen_layers):
        """
        Freeze the base model, except for the last few layers.
        
        Args:
            num_unfrozen_layers: Number of layers to unfreeze (from the end)
        """
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Then unfreeze the last n layers
        if num_unfrozen_layers > 0:
            try:
                # For different model architectures, we need to handle this differently
                if hasattr(self.model, 'transformer'):
                    # For GPT-Neo and similar models
                    layers = self.model.transformer.h
                    for i in range(1, num_unfrozen_layers + 1):
                        layer_idx = len(layers) - i
                        for param in layers[layer_idx].parameters():
                            param.requires_grad = True
                        print(f"Unfrozen layer {layer_idx} of {len(layers) - 1}")
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # For some newer models
                    layers = self.model.model.layers
                    for i in range(1, num_unfrozen_layers + 1):
                        layer_idx = len(layers) - i
                        for param in layers[layer_idx].parameters():
                            param.requires_grad = True
                        print(f"Unfrozen layer {layer_idx} of {len(layers) - 1}")
                else:
                    print("Unsupported model architecture. Could not unfreeze specific layers.")
                    # Unfreeze all parameters as a fallback
                    for param in self.model.parameters():
                        param.requires_grad = True