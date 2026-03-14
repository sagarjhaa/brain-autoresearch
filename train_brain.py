"""
Brain-AutoResearch: Training script optimized for brain-inspired efficiency patterns
Intel Mac compatible with CPU-only training and brain metrics
"""

import os
import gc
import math
import time
import platform
import psutil
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Brain metrics integration
from brain_metrics import BrainEfficiencyTracker, apply_brain_inspired_optimizations

# Intel Mac optimization setup
def setup_intel_mac():
    """Configure optimal settings for Intel Mac"""
    if platform.system() == "Darwin" and "Intel" in platform.processor():
        # CPU optimization
        cpu_count = os.cpu_count()
        torch.set_num_threads(max(1, cpu_count - 2))
        
        # Environment variables for Intel optimization
        os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_count - 2))
        os.environ["MKL_NUM_THREADS"] = str(max(1, cpu_count - 2))
        
        # Intel MKL optimization
        torch.backends.mkl.enabled = True
        
        print(f"Intel Mac detected: Using {torch.get_num_threads()} CPU threads")
        return torch.device("cpu")
    
    # GPU fallback
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps") 
    else:
        return torch.device("cpu")

# Setup device
device = setup_intel_mac()
print(f"Training on: {device}")

# Brain-optimized hyperparameters for Intel Mac
BRAIN_CONFIG = {
    # Smaller model for CPU training and brain-like efficiency
    "DEPTH": 4,                      # Reduced from 8 (brain-like shallow processing)
    "WIDTH": 384,                    # Smaller width for efficiency
    "TOTAL_BATCH_SIZE": 2**12,       # 4K tokens (reduced for CPU)
    "DEVICE_BATCH_SIZE": 16,         # Small batches for CPU
    "vocab_size": 4096,              # Smaller vocabulary
    "MAX_SEQ_LEN": 256,              # Shorter sequences for Intel Mac
    
    # Brain-inspired sparsity
    "TARGET_SPARSITY": 0.03,         # 3% activation (brain-like)
    "SPARSITY_WARMUP": 1000,         # Steps to reach target sparsity
    
    # Energy efficiency
    "ENERGY_WEIGHT": 0.1,            # Weight for energy in loss function
    "ADAPTIVE_PRECISION": True,      # Use mixed precision where beneficial
    
    # Hierarchical processing
    "LAYER_SPECIALIZATION": True,    # Different learning rates per layer
    "FORGETTING_RATE": 0.01,        # Active forgetting coefficient
}

try:
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
except ImportError:
    # Fallback constants for Intel Mac
    MAX_SEQ_LEN = BRAIN_CONFIG["MAX_SEQ_LEN"]
    TIME_BUDGET = 300  # 5 minutes
    print("Using brain-optimized fallback configuration")

# ---------------------------------------------------------------------------
# Brain-Inspired GPT Model
# ---------------------------------------------------------------------------

@dataclass  
class BrainGPTConfig:
    sequence_len: int = BRAIN_CONFIG["MAX_SEQ_LEN"]
    vocab_size: int = BRAIN_CONFIG["vocab_size"] 
    n_layer: int = BRAIN_CONFIG["DEPTH"]
    n_head: int = 4  # Reduced heads for efficiency
    n_kv_head: int = 4
    n_embd: int = BRAIN_CONFIG["WIDTH"]
    window_pattern: str = "L"  # Linear attention for CPU efficiency
    brain_sparsity: float = BRAIN_CONFIG["TARGET_SPARSITY"]


def norm(x):
    """Brain-inspired normalization (RMSNorm like biological homeostasis)"""
    return F.rms_norm(x, (x.size(-1),))


class SparseLinear(nn.Module):
    """Brain-inspired sparse linear layer with adaptive sparsity"""
    
    def __init__(self, in_features, out_features, sparsity=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sparsity = sparsity
        self.register_buffer('mask', torch.ones_like(self.linear.weight))
        
        # Initialize with sparse connectivity (brain-like)
        with torch.no_grad():
            sparse_mask = torch.rand_like(self.linear.weight) > sparsity
            self.mask.copy_(sparse_mask.float())
    
    def forward(self, x):
        # Apply sparse mask (brain-like connectivity)
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)
    
    def update_sparsity(self, target_sparsity):
        """Adapt sparsity based on usage (brain plasticity)"""
        with torch.no_grad():
            weight_importance = self.linear.weight.abs()
            threshold = torch.quantile(weight_importance.flatten(), target_sparsity)
            self.mask = (weight_importance > threshold).float()


class BrainAttention(nn.Module):
    """Brain-inspired sparse attention mechanism"""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # Brain-inspired sparse projections
        self.q_proj = SparseLinear(config.n_embd, config.n_embd, sparsity=0.1)
        self.k_proj = SparseLinear(config.n_embd, config.n_kv_head * self.head_size, sparsity=0.1)  
        self.v_proj = SparseLinear(config.n_embd, config.n_kv_head * self.head_size, sparsity=0.1)
        self.o_proj = SparseLinear(config.n_embd, config.n_embd, sparsity=0.1)
        
        self.sparsity_ratio = config.brain_sparsity
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Brain-inspired sparse projections
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size)  
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size)
        
        # CPU-efficient attention (no flash attention)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with sparse activation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        
        # Brain-inspired sparse attention (only attend to most relevant)
        if self.training and self.sparsity_ratio < 1.0:
            # Keep only top-k attention weights (sparse like brain)
            top_k = max(1, int(T * self.sparsity_ratio))
            att_sparse = torch.zeros_like(att)
            top_indices = torch.topk(att, top_k, dim=-1)[1]
            att_sparse.scatter_(-1, top_indices, att.gather(-1, top_indices))
            att = att_sparse
        
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)


class BrainMLP(nn.Module):
    """Brain-inspired sparse MLP with adaptive activation"""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2.5 * config.n_embd)  # Reduced from standard 4x
        
        # Sparse projections like brain synapses
        self.up_proj = SparseLinear(config.n_embd, hidden_dim, sparsity=0.15)
        self.gate_proj = SparseLinear(config.n_embd, hidden_dim, sparsity=0.15) 
        self.down_proj = SparseLinear(hidden_dim, config.n_embd, sparsity=0.15)
        
    def forward(self, x):
        # SwiGLU activation with sparse computation
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Brain-inspired sparse activation
        activated = F.silu(gate) * up
        
        return self.down_proj(activated)


class BrainTransformerBlock(nn.Module):
    """Brain-inspired transformer block with hierarchical processing"""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = BrainAttention(config)
        self.mlp = BrainMLP(config)
        
        # Brain-inspired adaptive processing
        self.layer_confidence = nn.Parameter(torch.ones(1))  # Confidence in this layer
        
    def forward(self, x):
        # Attention with residual (brain: cortical connections)
        attn_out = self.attention(norm(x))
        x = x + attn_out
        
        # MLP with residual (brain: subcortical processing)
        mlp_out = self.mlp(norm(x))
        x = x + mlp_out
        
        # Brain-inspired adaptive computation
        if self.training:
            # Layers learn confidence in their outputs (brain hierarchy)
            confidence = torch.sigmoid(self.layer_confidence)
            x = confidence * x + (1 - confidence) * x.detach()
        
        return x


class BrainGPT(nn.Module):
    """Brain-inspired GPT with biological efficiency patterns"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings (brain: input encoding)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.sequence_len, config.n_embd)
        
        # Brain-inspired transformer stack
        self.layers = nn.ModuleList([
            BrainTransformerBlock(config, i) for i in range(config.n_layer)
        ])
        
        # Output head (brain: motor output)
        self.output_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = SparseLinear(config.n_embd, config.vocab_size, sparsity=0.05)
        
        # Brain efficiency tracking
        self.brain_tracker = BrainEfficiencyTracker()
        
        # Apply brain-inspired initialization
        self.apply(self._init_weights)
        apply_brain_inspired_optimizations(self)
        
    def _init_weights(self, module):
        """Brain-inspired weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = token_emb + pos_emb
        
        # Brain-inspired hierarchical processing
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        # Calculate loss with brain-inspired efficiency terms
        if targets is not None:
            # Standard language modeling loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Brain-inspired efficiency regularization
            brain_metrics = self.brain_tracker.calculate_brain_metrics(self, idx[:1], T)
            
            # Energy efficiency term (encourage sparse computation)
            sparsity_loss = -brain_metrics.sparse_activation_ratio  # Negative to encourage sparsity
            energy_loss = brain_metrics.energy_per_token
            
            # Combined loss function
            total_loss = loss + 0.01 * sparsity_loss + 0.001 * energy_loss
            
            return logits, total_loss
        
        return logits, None
    
    def update_sparsity(self, step, warmup_steps):
        """Gradually increase sparsity during training (brain development)"""
        target_sparsity = self.config.brain_sparsity
        current_sparsity = min(target_sparsity, target_sparsity * step / warmup_steps)
        
        # Update all sparse layers
        for module in self.modules():
            if isinstance(module, SparseLinear):
                module.update_sparsity(current_sparsity)


# ---------------------------------------------------------------------------
# Brain-Inspired Training Loop  
# ---------------------------------------------------------------------------

def get_brain_optimizer(model, learning_rate=3e-4):
    """Brain-inspired optimizer with adaptive learning rates"""
    
    # Different learning rates for different layer types (brain: regional specialization)
    param_groups = []
    
    # Attention layers (brain: cortical processing)
    attention_params = []
    # MLP layers (brain: subcortical processing)  
    mlp_params = []
    # Other parameters
    other_params = []
    
    for name, param in model.named_parameters():
        if 'attention' in name or 'attn' in name:
            attention_params.append(param)
        elif 'mlp' in name or 'feed' in name:
            mlp_params.append(param)  
        else:
            other_params.append(param)
    
    # Brain-inspired learning rate hierarchy
    param_groups = [
        {'params': attention_params, 'lr': learning_rate * 1.2},      # Attention learns faster
        {'params': mlp_params, 'lr': learning_rate * 0.8},            # MLP learns slower  
        {'params': other_params, 'lr': learning_rate}                 # Standard rate
    ]
    
    # Use AdamW with brain-inspired parameters
    return torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))


def main():
    """Brain-AutoResearch main training loop"""
    
    # Initialize brain efficiency tracking
    brain_tracker = BrainEfficiencyTracker()
    brain_tracker.start_measurement()
    
    # Model setup
    config = BrainGPTConfig()
    model = BrainGPT(config).to(device)
    
    print(f"Brain Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Target Sparse Activation: {config.brain_sparsity:.1%}")
    print(f"Training Device: {device}")
    
    # Brain-inspired optimizer
    optimizer = get_brain_optimizer(model)
    
    try:
        # Data loading (fallback for Intel Mac)
        train_loader = make_dataloader("train", BRAIN_CONFIG["DEVICE_BATCH_SIZE"], device)
    except:
        print("Using dummy data for testing")
        # Dummy data for testing
        dummy_tokens = torch.randint(0, config.vocab_size, (BRAIN_CONFIG["DEVICE_BATCH_SIZE"], config.sequence_len))
        train_loader = [(dummy_tokens, dummy_tokens)]
    
    # Training loop with brain metrics
    model.train()
    step = 0
    tokens_processed = 0
    start_time = time.time()
    best_brain_score = 0.0
    
    print("Starting Brain-AutoResearch training...")
    print("Optimizing for brain-inspired efficiency patterns")
    
    while time.time() - start_time < TIME_BUDGET:
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if time.time() - start_time >= TIME_BUDGET:
                break
                
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass with brain efficiency tracking
            logits, loss = model(input_ids, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Brain-inspired gradient processing
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update brain-inspired sparsity
            model.update_sparsity(step, BRAIN_CONFIG["SPARSITY_WARMUP"])
            
            step += 1
            tokens_processed += input_ids.numel()
            
            # Brain metrics logging
            if step % 50 == 0:
                brain_metrics = brain_tracker.calculate_brain_metrics(model, input_ids[:1], tokens_processed)
                brain_score = brain_tracker.log_brain_metrics(brain_metrics, step)
                
                print(f"Step {step} | Loss: {loss.item():.4f} | Brain Score: {brain_score:.4f}")
                
                # Track best brain efficiency
                if brain_score > best_brain_score:
                    best_brain_score = brain_score
                    print(f"🧠 New best brain efficiency: {best_brain_score:.4f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("BRAIN-AUTORESEARCH RESULTS")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Get final brain metrics
        sample_input = torch.randint(0, config.vocab_size, (1, config.sequence_len)).to(device)
        final_metrics = brain_tracker.calculate_brain_metrics(model, sample_input, tokens_processed)
        final_brain_score = brain_tracker.get_brain_score(final_metrics)
        
        print(f"Training completed in {time.time() - start_time:.1f}s")
        print(f"Steps: {step}")
        print(f"Tokens processed: {tokens_processed:,}")
        print(f"Final Brain Efficiency Score: {final_brain_score:.4f}")
        print(f"Best Brain Efficiency Score: {best_brain_score:.4f}")
        print()
        print("Brain Efficiency Breakdown:")
        brain_tracker.log_brain_metrics(final_metrics, step)
        
        # Brain insights
        print("\n🧠 Brain-Inspired Insights Discovered:")
        if final_metrics.sparse_activation_ratio < 0.1:
            print(f"✓ Achieved sparse activation ({final_metrics.sparse_activation_ratio:.3f}) - brain-like efficiency!")
        if final_metrics.energy_per_token < 1.0:
            print(f"✓ Low energy consumption ({final_metrics.energy_per_token:.6f}) - approaching biological efficiency!")
        if final_metrics.neuron_death_ratio > 0.05:
            print(f"✓ Healthy neuron pruning ({final_metrics.neuron_death_ratio:.3f}) - natural selection at work!")
            
        print(f"\n🎯 Brain Efficiency Target Achievement: {final_brain_score/1.0*100:.1f}%")


if __name__ == "__main__":
    main()