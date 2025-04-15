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
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'  # Get device type as string
        self.use_half_precision = use_half_precision
        
        # Load the base LM with the appropriate precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_half_precision else None,
        ).to(self.device)
        
        # Add a value head with careful initialization
        self.v_head = nn.Linear(self.model.config.hidden_size, 1, bias=False).to(self.device)
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
            except Exception as e:
                print(f"Error unfreezing layers: {e}")
                # Unfreeze value head as a minimum
                for param in self.v_head.parameters():
                    param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Toxicity score values (lower = more toxic)
        """
        # Make sure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Use autocast for mixed precision if needed - use string device type
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_half_precision):
            # Get the hidden states from the base model
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use the last hidden state
            
            # Use mean pooling for more stable representations
            if attention_mask is not None:
                # Expand attention mask to match hidden state dimensions
                expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                # Apply mask and get sum
                masked_hidden = hidden_states * expanded_mask
                sum_hidden = torch.sum(masked_hidden, dim=1)
                # Get token count (avoid division by zero)
                token_count = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1.0)
                # Mean pooling
                pooled_hidden = sum_hidden / token_count
                # Apply value head and clamp values for stability
                values = self.v_head(pooled_hidden)
                values = torch.clamp(values, min=-10.0, max=10.0)
            else:
                # Fallback to last token if no mask
                last_token_indices = torch.tensor([input_ids.size(1)-1] * input_ids.size(0), device=input_ids.device)
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                last_hidden_states = hidden_states[batch_indices, last_token_indices]
                values = self.v_head(last_hidden_states)
                values = torch.clamp(values, min=-10.0, max=10.0)
                
            return values
    
    def save(self, path):
        """Save the model to disk."""
        # Save v_head and config
        state_dict = {
            'v_head': self.v_head.state_dict(),
            'config': {
                'model_name': self.model.config._name_or_path,
                'use_half_precision': self.use_half_precision
            }
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path, device="cuda"):
        """Load the model from disk."""
        state_dict = torch.load(path, map_location=device)
        config = state_dict['config']
        
        # Create a new model
        model = cls(config['model_name'], 
                   use_half_precision=config['use_half_precision'],
                   device=device, 
                   num_unfrozen_layers=0)  # Load with all frozen since we're just inferencing
        
        # Load v_head
        model.v_head.load_state_dict(state_dict['v_head'])
        
        return model