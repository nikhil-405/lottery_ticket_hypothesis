"""
Quick smoke test: Train with pruning on GPU for a few iterations
"""
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.lenet import LeNet300100
from utils.data_loader import get_mnist_dataloaders
from utils.trainer import Trainer
from utils.pruning import PruningManager

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load minimal data
    print("\nLoading MNIST data...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=60, fashion=False)
    
    # Create model
    print("Creating model...")
    model = LeNet300100().to(device)
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Create pruning manager
    print("\nInitializing pruning manager...")
    pruner = PruningManager(model)
    pruner.save_initial_weights()
    pruner.initialize_masks()
    
    # Check mask device
    modules = dict(model.named_modules())
    for name, buffer_name in list(pruner.masks.items())[:1]:  # Check first mask
        module_name, _, _ = name.rpartition('.')
        module = modules[module_name] if module_name else model
        mask = getattr(module, buffer_name)
        print(f"Mask '{name}' device: {mask.device}")
    
    # Do one round of pruning
    print("\nPruning 20% of weights...")
    sparsity = pruner.prune_by_magnitude(pruning_rate=0.2, layer_wise=True)
    print(f"Sparsity after pruning: {sparsity:.2f}%")
    
    # Train for a few iterations
    print("\nTraining for 200 iterations with pruning...")
    trainer = Trainer(model, device, train_loader, val_loader, test_loader)
    results = trainer.train(
        num_iterations=200,
        learning_rate=0.0012,
        pruner=pruner
    )
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Final test accuracy: {results['final_test_acc']:.2f}%")
    print(f"  Model device: {next(model.parameters()).device}")
    
    # Check masks still on correct device
    for name, buffer_name in list(pruner.masks.items())[:1]:
        module_name, _, _ = name.rpartition('.')
        module = modules[module_name] if module_name else model
        mask = getattr(module, buffer_name)
        print(f"  Mask '{name}' still on device: {mask.device}")
    
    print("\n✓ GPU training smoke test PASSED!")

if __name__ == "__main__":
    main()
