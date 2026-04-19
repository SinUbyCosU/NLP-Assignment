import torch
import torch.nn as nn

class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return x @ self.weight.T  # [batch, in] @ [in, out] → [batch, out]