import torch
import torch.nn as nn
import copy
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class PruningManager:
    """
    Manages iterative magnitude-based pruning for Lottery Ticket Hypothesis
    """
    def __init__(self, model):
        self.model = model
        self.initial_state = None
        self.masks = {}
        
    def save_initial_weights(self):
        """
        Save initial weights (θ₀) - the 'winning ticket' initialization
        Must be called before any training!
        """
        self.initial_state = copy.deepcopy(self.model.state_dict())
        print("✓ Initial weights saved")
    
    def initialize_masks(self):
        """
        Initialize masks to all ones (no pruning initially)
        Mask convention: 1 = keep weight, 0 = pruned
        """
        self.masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Only prune weights, not biases
                self.masks[name] = torch.ones_like(param.data)
        
        print(f"✓ Initialized masks for {len(self.masks)} weight tensors")
    
    def apply_masks(self):
        """
        Apply current masks to model parameters (zero out pruned weights)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    # Move mask to same device as parameter
                    mask = self.masks[name].to(param.device)
                    param.data *= mask

    def prune_by_magnitude(self, pruning_rate, layer_wise=True):
        """
        Prune weights by magnitude
        
        Args:
            pruning_rate: Fraction of remaining weights to prune (e.g., 0.2 = 20%)
            layer_wise: If True, prune each layer independently
                    If False, prune globally across all layers
        
        Returns:
            Current sparsity percentage
        """
        if not self.masks:
            self.initialize_masks()
        
        with torch.no_grad():
            if layer_wise:
                # Prune each layer separately
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        # Ensure mask is on same device as parameter
                        device = param.device
                        mask = self.masks[name].to(device)
                        
                        # Get currently active (non-zero) weights
                        active_weights = param.data[mask == 1]
                        
                        if len(active_weights) == 0:
                            continue
                        
                        # Calculate pruning threshold
                        num_to_prune = int(len(active_weights) * pruning_rate)
                        
                        if num_to_prune > 0:
                            # Find threshold (k-th smallest magnitude)
                            threshold = torch.topk(
                                active_weights.abs().flatten(),
                                num_to_prune,
                                largest=False
                            )[0].max()
                            
                            # Update mask: prune weights below threshold
                            new_mask = (param.data.abs() > threshold).float()
                            
                            # Combine with existing mask (once pruned, stays pruned)
                            self.masks[name] = torch.min(mask, new_mask).cpu()  # Store on CPU
            
            else:
                # Global pruning across all layers
                all_weights = []
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        mask = self.masks[name].to(param.device)
                        active_weights = param.data[mask == 1]
                        all_weights.append(active_weights.flatten())
                
                # Concatenate all active weights
                all_weights = torch.cat(all_weights)
                
                # Calculate global threshold
                num_to_prune = int(len(all_weights) * pruning_rate)
                
                if num_to_prune > 0:
                    threshold = torch.topk(
                        all_weights.abs(),
                        num_to_prune,
                        largest=False
                    )[0].max()
                    
                    # Update masks globally
                    for name, param in self.model.named_parameters():
                        if name in self.masks:
                            new_mask = (param.data.abs() > threshold).float()
                            mask = self.masks[name].to(param.device)
                            self.masks[name] = torch.min(mask, new_mask).cpu()  # Store on CPU
            
            # Apply masks
            self.apply_masks()
        
        # Calculate and return current sparsity
        sparsity = self.get_sparsity()
        return sparsity   
     
    def reset_to_initial_weights(self):
        """
        Reset remaining (non-pruned) weights to their initial values (θ₀)
        This is the KEY step in Lottery Ticket Hypothesis!
        """
        if self.initial_state is None:
            raise ValueError("Initial weights not saved! Call save_initial_weights() first.")
        
        with torch.no_grad():
            # Get the device of the first model parameter
            device = next(self.model.parameters()).device
            
            # Load initial weights
            initial_state_device = {k: v.to(device) for k, v in self.initial_state.items()}
            self.model.load_state_dict(initial_state_device)
            
            # Re-apply masks (zero out pruned weights)
            self.apply_masks()
    
    print("✓ Weights reset to initial values (with current mask applied)")   
    def get_sparsity(self):
        """Calculate current sparsity percentage"""
        total_params = 0
        pruned_params = 0
        
        for name, mask in self.masks.items():
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
        
        sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
        return sparsity
    
    def get_remaining_percentage(self):
        """Get percentage of weights remaining"""
        return 100.0 - self.get_sparsity()
    
    def print_pruning_stats(self):
        """Print detailed pruning statistics"""
        print(f"\n{'='*60}")
        print("Pruning Statistics:")
        print(f"{'='*60}")
        
        for name, mask in self.masks.items():
            total = mask.numel()
            remaining = (mask == 1).sum().item()
            pruned = total - remaining
            remaining_pct = 100.0 * remaining / total
            
            print(f"  {name:20s}: {remaining:6d}/{total:6d} ({remaining_pct:5.1f}% remaining)")
        
        total_params = sum(mask.numel() for mask in self.masks.values())
        total_remaining = sum((mask == 1).sum().item() for mask in self.masks.values())
        overall_sparsity = self.get_sparsity()
        
        print(f"{'='*60}")
        print(f"  Overall: {total_remaining:,}/{total_params:,} ({100-overall_sparsity:.2f}% remaining)")
        print(f"  Sparsity: {overall_sparsity:.2f}%")
        print(f"{'='*60}\n")


# Test pruning
if __name__ == "__main__":
    from models.lenet import LeNet300100
    
    print("Testing Pruning Manager\n")
    
    # Create model
    model = LeNet300100()
    print(f"Initial parameters: {model.count_parameters():,}")
    
    # Create pruning manager
    pruner = PruningManager(model)
    
    # Save initial weights
    pruner.save_initial_weights()
    
    # Initialize masks
    pruner.initialize_masks()
    
    # Simulate iterative pruning
    print("\nSimulating Iterative Pruning (20% per round):\n")
    
    for round in range(5):
        sparsity = pruner.prune_by_magnitude(pruning_rate=0.2, layer_wise=True)
        print(f"Round {round + 1}: Sparsity = {sparsity:.2f}%, Remaining = {pruner.get_remaining_percentage():.2f}%")
    
    # Print detailed stats
    pruner.print_pruning_stats()
    
    # Test reset
    print("Testing weight reset...")
    pruner.reset_to_initial_weights()
    print(f"Non-zero parameters after reset: {model.count_nonzero_parameters():,}")