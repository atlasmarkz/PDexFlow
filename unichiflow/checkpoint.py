import torch
import torch.nn as nn
from typing import Dict, Any

def load_backbone_into_lora(model: nn.Module, backbone_state_dict: Dict[str, Any], verbose: bool = True):
    """
    Loads a backbone state dict (without LoRA layers) into a model with LoRA layers.
    Automatically maps 'layer.weight' to 'layer.base.weight' for ChiLoRALinear modules.
    """
    model_state = model.state_dict()
    new_state_dict = {}
    
    # Get mapping of current model keys
    # We need to know which keys in the model correspond to ChiLoRALinear base layers
    
    # Naive approach: iterate over backbone keys and try to match them to model keys
    # If key exists in model, use it.
    # If key + ".base.weight" exists in model, use that.
    
    loaded_keys = []
    skipped_keys = []
    
    for k, v in backbone_state_dict.items():
        if k in model_state:
            new_state_dict[k] = v
            loaded_keys.append(k)
        elif k.endswith(".weight"):
            # Try mapping to .base.weight
            prefix = k[:-7] # remove .weight
            lora_key = prefix + ".base.weight"
            if lora_key in model_state:
                new_state_dict[lora_key] = v
                loaded_keys.append(lora_key)
            else:
                skipped_keys.append(k)
        elif k.endswith(".bias"):
            # Try mapping to .base.bias
            prefix = k[:-5] # remove .bias
            lora_key = prefix + ".base.bias"
            if lora_key in model_state:
                new_state_dict[lora_key] = v
                loaded_keys.append(lora_key)
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)
            
    # Load the constructed state dict
    # strict=False because we are missing LoRA weights (A and B) which will be initialized randomly
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if verbose:
        print(f"Loaded {len(loaded_keys)} keys from backbone.")
        print(f"Skipped {len(skipped_keys)} keys from backbone (not found in LoRA model).")
        # Filter out LoRA params from missing list to see if we missed any backbone params
        real_missing = [k for k in missing if "lora_" not in k and ".base." not in k]
        if real_missing:
            print(f"WARNING: The following backbone parameters were NOT loaded: {real_missing}")
        else:
            print("All backbone parameters loaded successfully.")

    return model
