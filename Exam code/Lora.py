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
class LoRAAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, rank=8):
        super().__init__()
        
        # Standard projections (frozen)
        self.W_q = LoRALinear(dim, dim, rank=rank)
        self.W_k = nn.Linear(dim, dim)  # No LoRA (frozen entirely)
        self.W_v = LoRALinear(dim, dim, rank=rank)
        self.W_o = nn.Linear(dim, dim)  # No LoRA (frozen entirely)
        
        # Freeze non-LoRA weights
        for param in self.W_k.parameters():
            param.requires_grad = False
        for param in self.W_o.parameters():
            param.requires_grad = False
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Projections (only Q and V use LoRA)
        Q = self.W_q(x)  # Uses LoRA
        K = self.W_k(x)  # Frozen
        V = self.W_v(x)  # Uses LoRA
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Attention computation
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bvhd->bqhd', attn_weights, V)
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, dim)
        
        # Output projection (frozen)
        return self.W_o(output)
# lora_B.grad = [values]
def merge_lora_weights(layer):
    """
    Merge LoRA weights into base weights for deployment
    """
    # Compute the low-rank update
    delta_W = layer.lora_A @ layer.lora_B  # [in, rank] @ [rank, out]
    
    # Add to frozen weights
    merged_weight = layer.weight + (layer.scaling * delta_W.T)
    
    # Create new standard layer
    new_layer = nn.Linear(layer.weight.shape[1], layer.weight.shape[0])
    new_layer.weight.data = merged_weight
    
    return new_layer