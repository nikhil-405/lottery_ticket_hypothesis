import torch
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lenet import LeNet300100
from utils.data_loader import get_mnist_dataloaders
from utils.trainer import Trainer
from utils.pruning import PruningManager

class LotteryTicketExperiment:
    """
    Main experiment for Lottery Ticket Hypothesis
    Implements iterative magnitude pruning with weight resetting
    """
    
    def __init__(self, 
                 pruning_rate=0.2,
                 num_rounds=15,
                 training_iterations=50000,
                 learning_rate=0.0012,
                 batch_size=60,
                 device=None,
                 fashion=True):
        """
        Args:
            pruning_rate: Fraction of weights to prune each round (default: 20%)
            num_rounds: Number of pruning rounds (default: 15, reaches ~3.5% remaining)
            training_iterations: Training iterations per round (default: 50K)
            learning_rate: Adam learning rate (default: 0.0012)
            batch_size: Batch size (default: 60)
            device: Device to use (auto-detect if None)
            fashion: Use Fashion-MNIST if True, MNIST if False (default: True)
        """
        self.pruning_rate = pruning_rate
        self.num_rounds = num_rounds
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fashion = fashion
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Results storage
        self.results = {
            'config': {
                'pruning_rate': pruning_rate,
                'num_rounds': num_rounds,
                'training_iterations': training_iterations,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'device': str(self.device)
            },
            'rounds': []
        }
        
        print(f"\n{'='*70}")
        print(f"LOTTERY TICKET HYPOTHESIS EXPERIMENT")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Pruning rate per round: {pruning_rate*100}%")
        print(f"  Number of rounds: {num_rounds}")
        print(f"  Training iterations per round: {training_iterations:,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
    
    def run(self):
        """Run the complete iterative pruning experiment"""
        
        # Load data
        dataset_name = "Fashion-MNIST" if self.fashion else "MNIST"
        print(f"Loading {dataset_name}...")
        train_loader, val_loader, test_loader = get_mnist_dataloaders(
            batch_size=self.batch_size,
            fashion=self.fashion
        )
        
        # Create model
        print("Initializing model...")
        model = LeNet300100()
        
        # Create pruning manager and save initial weights
        pruner = PruningManager(model)
        pruner.save_initial_weights()
        pruner.initialize_masks()
        
        # Run iterative pruning
        for round_num in range(self.num_rounds):
            print(f"\n{'='*70}")
            print(f"ROUND {round_num + 1}/{self.num_rounds}")
            print(f"{'='*70}")
            
            round_start_time = time.time()
            
            # Get current sparsity
            sparsity = pruner.get_sparsity()
            remaining_pct = pruner.get_remaining_percentage()
            
            print(f"Starting sparsity: {sparsity:.2f}% ({remaining_pct:.2f}% weights remaining)")
            
            # Create trainer
            trainer = Trainer(model, self.device, train_loader, val_loader, test_loader)
            
            # Train
            train_results = trainer.train(
                num_iterations=self.training_iterations,
                learning_rate=self.learning_rate,
                pruner=pruner
            )
            
            # Get early stopping iteration
            early_stop_iter = trainer.get_early_stopping_iteration()
            early_stop_idx = trainer.history['iteration'].index(early_stop_iter)
            early_stop_val_acc = trainer.history['val_acc'][early_stop_idx]
            
            # Store results for this round
            round_results = {
                'round': round_num + 1,
                'sparsity_before_pruning': sparsity,
                'remaining_pct_before_pruning': remaining_pct,
                'best_val_loss': train_results['best_val_loss'],
                'best_val_iteration': train_results['best_val_iteration'],
                'early_stop_val_acc': early_stop_val_acc,
                'final_test_acc': train_results['final_test_acc'],
                'training_time': time.time() - round_start_time,
                'history': train_results['history']
            }
            
            print(f"\nRound {round_num + 1} Results:")
            print(f"  Early-stop iteration: {early_stop_iter:,}")
            print(f"  Early-stop val accuracy: {early_stop_val_acc:.2f}%")
            print(f"  Final test accuracy: {train_results['final_test_acc']:.2f}%")
            print(f"  Training time: {round_results['training_time']:.1f}s")
            
            # Prune for next round (except on last round)
            if round_num < self.num_rounds - 1:
                print(f"\nPruning {self.pruning_rate*100}% of remaining weights...")
                new_sparsity = pruner.prune_by_magnitude(
                    pruning_rate=self.pruning_rate,
                    layer_wise=True
                )
                
                round_results['sparsity_after_pruning'] = new_sparsity
                round_results['remaining_pct_after_pruning'] = 100 - new_sparsity
                
                print(f"New sparsity: {new_sparsity:.2f}% ({100-new_sparsity:.2f}% remaining)")
                
                # Reset to initial weights
                pruner.reset_to_initial_weights()
            
            self.results['rounds'].append(round_results)
        
        # Print final summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all rounds"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"{'Round':<8} {'Remaining%':<12} {'Early-Stop':<12} {'Val Acc%':<10} {'Test Acc%':<10}")
        print(f"{'-'*70}")
        
        for r in self.results['rounds']:
            print(f"{r['round']:<8} "
                  f"{r['remaining_pct_before_pruning']:<12.2f} "
                  f"{r['best_val_iteration']:<12,} "
                  f"{r['early_stop_val_acc']:<10.2f} "
                  f"{r['final_test_acc']:<10.2f}")
        
        print(f"{'='*70}\n")
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/lottery_ticket_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Results saved to {filename}")
        
        return filename


def main():
    """Run a quick test experiment (3 rounds, 5K iterations each)"""
    
    experiment = LotteryTicketExperiment(
        pruning_rate=0.2,
        num_rounds=3,
        training_iterations=5000,  # Quick test
        learning_rate=0.0012,
        batch_size=60
    )
    
    results = experiment.run()
    experiment.save_results()


if __name__ == "__main__":
    main()