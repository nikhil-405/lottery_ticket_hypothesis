import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.iterative_pruning import LotteryTicketExperiment
from utils.pruning import PruningManager
from models.lenet import LeNet300100
from utils.data_loader import get_mnist_dataloaders
from utils.trainer import Trainer
import time

class RandomReinitExperiment(LotteryTicketExperiment):
    """
    Same as lottery ticket but with random reinitialization after pruning
    This should perform WORSE, proving initialization matters!
    """
    
    def run(self):
        """Run iterative pruning with random reinitialization"""
        
        print("\nRANDOM REINITIALIZATION EXPERIMENT")
        print("Weights will be randomly reinitialized after each pruning round\n")
        
        # Load data
        train_loader, val_loader, test_loader = get_mnist_dataloaders(
            batch_size=self.batch_size
        )
        
        # Create model
        model = LeNet300100()
        
        # Create pruning manager
        pruner = PruningManager(model)
        pruner.save_initial_weights()  # Save for reference, but won't use
        pruner.initialize_masks()
        
        for round_num in range(self.num_rounds):
            print(f"\n{'='*70}")
            print(f"ROUND {round_num + 1}/{self.num_rounds}")
            print(f"{'='*70}")
            
            round_start_time = time.time()
            sparsity = pruner.get_sparsity()
            remaining_pct = pruner.get_remaining_percentage()
            
            print(f"Starting sparsity: {sparsity:.2f}% ({remaining_pct:.2f}% remaining)")
            
            # Train
            trainer = Trainer(model, self.device, train_loader, val_loader, test_loader)
            train_results = trainer.train(
                num_iterations=self.training_iterations,
                learning_rate=self.learning_rate
            )
            
            early_stop_iter = trainer.get_early_stopping_iteration()
            early_stop_idx = trainer.history['iteration'].index(early_stop_iter)
            early_stop_val_acc = trainer.history['val_acc'][early_stop_idx]
            
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
            
            # Prune and RANDOMLY REINITIALIZE
            if round_num < self.num_rounds - 1:
                new_sparsity = pruner.prune_by_magnitude(
                    pruning_rate=self.pruning_rate,
                    layer_wise=True
                )
                
                round_results['sparsity_after_pruning'] = new_sparsity
                round_results['remaining_pct_after_pruning'] = 100 - new_sparsity
                
                print(f"\nRANDOM REINITIALIZATION (not resetting to θ₀)")
                
                # Reinitialize the model with NEW random weights
                model = LeNet300100()
                model.to(self.device)
                
                # Apply the SAME mask to the new random weights
                for name, param in model.named_parameters():
                    if name in pruner.masks:
                        param.data *= pruner.masks[name].to(self.device)
            
            self.results['rounds'].append(round_results)
        
        self.print_summary()
        return self.results


def main():
    experiment = RandomReinitExperiment(
        pruning_rate=0.2,
        num_rounds=15,
        training_iterations=50000,
        learning_rate=0.0012,
        batch_size=60
    )
    
    results = experiment.run()
    experiment.save_results('results/random_reinit_experiment.json')


if __name__ == "__main__":
    main()