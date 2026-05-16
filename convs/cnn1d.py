import torch
import torch.nn as nn

class CNN1DFeatureExtractor(nn.Module):
    """
    1-D CNN for features extraction from NetFlow data.
    Architecture:
    Conv1d (k3, p0, s1) x 2 -> MaxPool1d (k2, s2) -> Conv1d (k3, p0, s1) x 2 -> AdaptiveMaxPool1d
    """

    def __init__(self, input_dim=41, output_dim=64):
        """
        Args:
            input_dim: Number of features (default 41)
            output_dim: Feature embedding dimension
        """
        super(CNN1DFeatureExtractor, self).__init__()
        
        # Architecture: 4 Conv1d layers (kernel 3, stride 1, padding 0)
        self.body = nn.Sequential(
            # Lớp 1
            nn.Conv1d(1, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            
            # Lớp 2
            nn.Conv1d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Lớp 3
            nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Lớp 4
            nn.Conv1d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Global Pooling
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        out = self.body(x)
        out = out.view(out.size(0), -1) # Flatten (64 dimensions)
        return out


class CNN1DConvNet(nn.Module):
    """
    Wrapper cho CNN1DFeatureExtractor, tương thích với SPCIL.
    
    Input:  tensor (batch_size, 31)   — tabular features
    Output: dict {"features": tensor(batch_size, 64), "fmaps": []}
    """
    out_dim = 64  # Output dimension của CNN1DFeatureExtractor

    def __init__(self):
        super(CNN1DConvNet, self).__init__()
        self.cnn = CNN1DFeatureExtractor(input_dim=31, output_dim=self.out_dim)

    def forward(self, x):
        """
        Args:
            x: tensor (B, 31)
        Returns:
            dict với "features" (B, 64) và "fmaps" (list rỗng)
        """
        features = self.cnn(x)  # (B, 64)
        return {"features": features, "fmaps": []}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
