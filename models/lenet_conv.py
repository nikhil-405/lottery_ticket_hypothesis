import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    Convolutional LeNet-5 architecture for MNIST
    Original LeCun et al. 1998 architecture
    
    Architecture:
    - Conv1: 1x28x28 -> 6x24x24 (5x5 kernel)
    - Pool1: 6x24x24 -> 6x12x12 (2x2 maxpool)
    - Conv2: 6x12x12 -> 16x8x8 (5x5 kernel)
    - Pool2: 16x8x8 -> 16x4x4 (2x2 maxpool)
    - FC1: 256 -> 120
    - FC2: 120 -> 84
    - FC3: 84 -> 10
    
    Total parameters: ~61K
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 channels * 4x4 after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Gaussian Glorot (Xavier) initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Conv1 + ReLU + MaxPool
        x = F.relu(self.conv1(x))      # (batch, 6, 24, 24)
        x = F.max_pool2d(x, 2)         # (batch, 6, 12, 12)
        
        # Conv2 + ReLU + MaxPool
        x = F.relu(self.conv2(x))      # (batch, 16, 8, 8)
        x = F.max_pool2d(x, 2)         # (batch, 16, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)      # (batch, 256)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))        # (batch, 120)
        x = F.relu(self.fc2(x))        # (batch, 84)
        x = self.fc3(x)                # (batch, 10)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_nonzero_parameters(self):
        """Count non-zero parameters (for sparsity calculation)"""
        return sum((p != 0).sum().item() for p in self.parameters())
    
    def get_sparsity(self):
        """Calculate current sparsity percentage"""
        total = self.count_parameters()
        nonzero = self.count_nonzero_parameters()
        return 100.0 * (total - nonzero) / total


# Test the model
if __name__ == "__main__":
    model = LeNet5()
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Architecture breakdown:")
    print(f"  Conv1 (1->6):   {6*5*5 + 6:,} params")
    print(f"  Conv2 (6->16):  {16*6*5*5 + 16:,} params")
    print(f"  FC1 (256->120): {256*120 + 120:,} params")
    print(f"  FC2 (120->84):  {120*84 + 84:,} params")
    print(f"  FC3 (84->10):   {84*10 + 10:,} params")
    
    # Test forward pass
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Current sparsity: {model.get_sparsity():.2f}%")