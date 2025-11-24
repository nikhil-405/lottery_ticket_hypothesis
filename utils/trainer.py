import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Trainer:
    """
    Trainer class for neural network training and evaluation
    """
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': [],
            'iteration': []
        }
    
    def train_epoch(self, optimizer, pruner=None):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # CRITICAL: Reapply mask after optimizer step to keep pruned weights at zero
            if pruner is not None:
                pruner.apply_masks()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, data_loader, desc='Evaluating'):
        """Evaluate on given data loader"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_iterations, learning_rate=0.0012, eval_every=100, pruner=None):
        """
        Train for specified number of iterations
        
        Args:
            num_iterations: Total training iterations (paper uses 50,000)
            learning_rate: Learning rate for Adam optimizer
            eval_every: Evaluate every N iterations
            pruner: PruningManager to apply masks during training
        """
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Calculate iterations per epoch
        iterations_per_epoch = len(self.train_loader)
        num_epochs = (num_iterations + iterations_per_epoch - 1) // iterations_per_epoch
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Total iterations: {num_iterations:,}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Optimizer: Adam")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Non-zero parameters: {self.model.count_nonzero_parameters():,}")
        print(f"  Sparsity: {self.model.get_sparsity():.2f}%")
        print(f"{'='*60}\n")
        
        iteration = 0
        best_val_loss = float('inf')
        best_val_iteration = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch(optimizer, pruner=pruner)
            iteration += len(self.train_loader)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader, desc='Validation')
            
            # Track best validation loss (for early stopping criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_iteration = iteration
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['iteration'].append(iteration)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Loss: {best_val_loss:.4f} at iteration {best_val_iteration}")
            
            # Stop if we've exceeded target iterations
            if iteration >= num_iterations:
                break
        
        # Final test evaluation
        test_loss, test_acc = self.evaluate(self.test_loader, desc='Test')
        self.history['test_acc'].append(test_acc)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Final Test Accuracy: {test_acc:.2f}%")
        print(f"  Best Validation Loss: {best_val_loss:.4f} at iteration {best_val_iteration}")
        print(f"{'='*60}\n")
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_iteration': best_val_iteration,
            'final_test_acc': test_acc,
            'history': self.history
        }
    
    def get_early_stopping_iteration(self):
        """Get iteration with minimum validation loss (early stopping criterion)"""
        min_val_loss_idx = self.history['val_loss'].index(min(self.history['val_loss']))
        return self.history['iteration'][min_val_loss_idx]


if __name__ == "__main__":
    from models.lenet import LeNet300100
    from utils.data_loader import get_mnist_dataloaders
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=60)
    
    # Create model
    model = LeNet300100()
    
    # Create trainer
    trainer = Trainer(model, device, train_loader, val_loader, test_loader)
    
    # Train for 2 epochs (quick test - paper uses 50K iterations ≈ 54 epochs)
    results = trainer.train(num_iterations=2000, learning_rate=0.0012)
    
    print(f"\nEarly stopping would occur at iteration: {trainer.get_early_stopping_iteration()}")