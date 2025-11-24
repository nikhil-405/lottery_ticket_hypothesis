import torch
import torch.nn as nn
import copy
from datetime import datetime
from pathlib import Path

class PruningManager:
    """
    Manages iterative magnitude-based pruning for Lottery Ticket Hypothesis
    """
    def __init__(self, model, log_file=None):
        self.model = model
        self.initial_state = None
        self.masks = {}
        
        # Setup logging
        self.log_file = log_file
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Create/clear log file and write header
            with open(self.log_file, 'w') as f:
                f.write(f"Pruning Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
    
    def log(self, message):
        """Print message and save to log file if specified"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        
    def save_initial_weights(self):
        """
        Save initial weights (θ₀) - the 'winning ticket' initialization
        Must be called before any training!
        """
        self.initial_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.log("✓ Initial weights saved")
    
    def initialize_masks(self):
        """
        Initialize masks to all ones (no pruning initially)
        Mask convention: 1 = keep weight, 0 = pruned
        """
        self.masks = {}
        for name, param in self.model.named_parameters():
            # Prune both conv and fc weights, but not biases
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param.data).cpu()
    
        self.log(f"✓ Initialized masks for {len(self.masks)} weight tensors")
    
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
                        # Get device of parameter
                        device = param.device
                        
                        # Move mask to device temporarily for computation
                        mask = self.masks[name].to(device)
                        
                        # Get currently active (non-zero) weights
                        active_mask = (mask == 1)
                        active_weights = param.data[active_mask]
                        
                        if len(active_weights) == 0:
                            continue
                        
                        # Calculate pruning threshold
                        num_to_prune = int(len(active_weights) * pruning_rate)
                        
                        if num_to_prune > 0:
                            # Find threshold (k-th smallest magnitude)
                            # Make sure active_weights is on the same device
                            abs_weights = active_weights.abs().flatten()
                            
                            # Use kthvalue instead of topk for better stability
                            if num_to_prune < len(abs_weights):
                                threshold = torch.kthvalue(abs_weights, num_to_prune)[0]
                            else:
                                threshold = abs_weights.max()
                            
                            # Update mask: prune weights below or equal to threshold
                            new_mask = (param.data.abs() > threshold).float()
                            
                            # Combine with existing mask (once pruned, stays pruned)
                            combined_mask = torch.min(mask, new_mask)
                            
                            # Store mask back on CPU
                            self.masks[name] = combined_mask.cpu()
            
            else:
                # Global pruning across all layers
                all_weights = []
                all_names = []
                
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        device = param.device
                        mask = self.masks[name].to(device)
                        active_weights = param.data[mask == 1]
                        all_weights.append(active_weights.flatten())
                        all_names.append(name)
                
                # Concatenate all active weights
                if len(all_weights) > 0:
                    all_weights = torch.cat(all_weights)
                    
                    # Calculate global threshold
                    num_to_prune = int(len(all_weights) * pruning_rate)
                    
                    if num_to_prune > 0 and num_to_prune < len(all_weights):
                        abs_weights = all_weights.abs()
                        threshold = torch.kthvalue(abs_weights, num_to_prune)[0]
                        
                        # Update masks globally
                        for name, param in self.model.named_parameters():
                            if name in self.masks:
                                device = param.device
                                mask = self.masks[name].to(device)
                                new_mask = (param.data.abs() > threshold).float()
                                combined_mask = torch.min(mask, new_mask)
                                self.masks[name] = combined_mask.cpu()
            
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
            
            # Load initial weights to the correct device
            initial_state_device = {k: v.to(device) for k, v in self.initial_state.items()}
            self.model.load_state_dict(initial_state_device)
            
            # Re-apply masks (zero out pruned weights)
            self.apply_masks()
        
        self.log("✓ Weights reset to initial values (with current mask applied)")
    
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
        self.log(f"\n{'='*60}")
        self.log("Pruning Statistics:")
        self.log(f"{'='*60}")
        
        for name, mask in self.masks.items():
            total = mask.numel()
            remaining = (mask == 1).sum().item()
            pruned = total - remaining
            remaining_pct = 100.0 * remaining / total
            
            self.log(f"  {name:20s}: {remaining:6d}/{total:6d} ({remaining_pct:5.1f}% remaining)")
        
        total_params = sum(mask.numel() for mask in self.masks.values())
        total_remaining = sum((mask == 1).sum().item() for mask in self.masks.values())
        overall_sparsity = self.get_sparsity()
        
        self.log(f"{'='*60}")
        self.log(f"  Overall: {total_remaining:,}/{total_params:,} ({100-overall_sparsity:.2f}% remaining)")
        self.log(f"  Sparsity: {overall_sparsity:.2f}%")
        self.log(f"{'='*60}\n")


# Test pruning WITH GPU
if __name__ == "__main__":
    from models.lenet import LeNet300100
    
    print("Testing Pruning Manager with GPU simulation\n")
    
    # Test on both CPU and GPU
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device_name in devices:
        print(f"\n{'='*60}")
        print(f"Testing on {device_name.upper()}")
        print(f"{'='*60}\n")
        
        device = torch.device(device_name)
        
        # Create model and move to device
        model = LeNet300100().to(device)
        print(f"Model on device: {next(model.parameters()).device}")
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
        print(f"Model still on device: {next(model.parameters()).device}\n")