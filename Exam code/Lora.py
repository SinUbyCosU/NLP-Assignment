import torch
import torch.nn as nn

class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return x @ self.weight.T  # [batch, in] @ [in, out] → [batch, out]

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # Frozen pre-trained weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # FREEZE!
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Standard path (frozen)
        result = x @ self.weight.T
        
        # LoRA path (trainable)
        lora_result = (x @ self.lora_A) @ self.lora_B
        
        # Combine
        return result + self.scaling * lora_result

# Create layer
layer = LoRALinear(in_features=768, out_features=768, rank=8)

# Input
x = torch.randn(32, 768)  # batch_size=32, dim=768

# Forward pass
output = layer(x)  # [32, 768]

# During training
loss = compute_loss(output, target)
loss.backward()

# Only lora_A and lora_B get gradients!
# weight.grad = None (frozen)
# lora_A.grad = [values]
# lora_B.grad = [values]