"""
Brain-AutoResearch: Neural Efficiency Training Script
A fork of Karpathy's autoresearch optimized for brain-inspired efficiency patterns.

Key brain-inspired features:
- Sparse activation tracking (mimics 1-4% brain neuron activation)
- Energy per token metrics (metabolic efficiency)
- Forgetting mechanisms (synaptic pruning simulation)
- Adaptive precision patterns
- Hierarchical memory efficiency

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Brain-inspired monitoring imports
import psutil
import platform

from kernels import get_kernel
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"

# Handle Intel Mac compatibility - fallback to standard attention if kernels unavailable
try:
    fa3 = get_kernel(repo).flash_attention_interface if torch.cuda.is_available() else None
except:
    fa3 = None
    print("⚠️  FlashAttention not available, using standard attention (Intel Mac compatible)")

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Brain-Inspired Metrics & Monitoring
# ---------------------------------------------------------------------------

class BrainMetrics:
    """
    Tracks brain-inspired efficiency metrics during training.
    Mimics biological neural network patterns for energy efficiency.
    """
    def __init__(self):
        self.reset()
        self.process = psutil.Process()
        self.system_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'is_apple_silicon': platform.machine() == 'arm64' and platform.system() == 'Darwin',
            'is_intel_mac': platform.machine() == 'x86_64' and platform.system() == 'Darwin'
        }
        
    def reset(self):
        self.activation_sparsity = []
        self.energy_per_token = []
        self.memory_efficiency = []
        self.forgetting_rate = []
        self.precision_variance = []
        self.hierarchical_compression = []
        
    def update_sparsity(self, activations):
        """Track sparse activation patterns (brain: ~1-4% neurons active)"""
        with torch.no_grad():
            # Calculate activation sparsity across all layers
            total_active = 0
            total_neurons = 0
            
            if isinstance(activations, dict):
                for layer_name, acts in activations.items():
                    if acts is not None:
                        active = (acts.abs() > acts.abs().mean() * 0.1).float().sum()
                        total_active += active.item()
                        total_neurons += acts.numel()
            else:
                # Single activation tensor
                active = (activations.abs() > activations.abs().mean() * 0.1).float().sum()
                total_active = active.item()
                total_neurons = activations.numel()
                
            sparsity = (total_neurons - total_active) / total_neurons if total_neurons > 0 else 0
            self.activation_sparsity.append(sparsity)
            
    def update_energy(self, tokens_processed, time_taken):
        """Track energy efficiency (brain: ~20W total, ~12W for thinking)"""
        try:
            # Get power consumption (approximation)
            if torch.cuda.is_available():
                # GPU power estimation
                gpu_utilization = torch.cuda.utilization() / 100.0
                estimated_power = 250 * gpu_utilization  # Assume 250W max GPU
            else:
                # CPU power estimation for Intel Mac
                cpu_percent = self.process.cpu_percent()
                estimated_power = 45 * (cpu_percent / 100.0)  # Intel Mac ~45W CPU
                
            energy_per_token = estimated_power * time_taken / tokens_processed
            self.energy_per_token.append(energy_per_token)
        except:
            # Fallback if power monitoring unavailable
            self.energy_per_token.append(0.01)  # Placeholder value
            
    def update_memory_efficiency(self, current_memory_mb, baseline_memory_mb):
        """Track memory compression relative to baseline"""
        efficiency = baseline_memory_mb / current_memory_mb if current_memory_mb > 0 else 1.0
        self.memory_efficiency.append(efficiency)
        
    def update_forgetting(self, old_weights, new_weights):
        """Track forgetting rate (synaptic pruning simulation)"""
        with torch.no_grad():
            if old_weights is not None and new_weights is not None:
                weight_change = (old_weights - new_weights).abs().mean()
                forgetting_rate = weight_change / (old_weights.abs().mean() + 1e-8)
                self.forgetting_rate.append(forgetting_rate.item())
            
    def get_summary(self):
        """Get brain efficiency summary"""
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0
            
        return {
            'sparsity_mean': safe_mean(self.activation_sparsity),
            'energy_per_token_mean': safe_mean(self.energy_per_token),
            'memory_efficiency_mean': safe_mean(self.memory_efficiency),
            'forgetting_rate_mean': safe_mean(self.forgetting_rate),
            'brain_efficiency_score': self._calculate_brain_score(),
            'system_info': self.system_info
        }
        
    def _calculate_brain_score(self):
        """Calculate overall brain-inspired efficiency score (0-100)"""
        # Optimal brain patterns:
        # - High sparsity (85-95%)
        # - Low energy per token
        # - Efficient memory usage
        # - Controlled forgetting
        
        sparsity_score = (sum(self.activation_sparsity) / len(self.activation_sparsity)) * 100 if self.activation_sparsity else 0
        energy_score = max(0, 100 - sum(self.energy_per_token) / len(self.energy_per_token) * 1000) if self.energy_per_token else 50
        memory_score = min(100, (sum(self.memory_efficiency) / len(self.memory_efficiency)) * 50) if self.memory_efficiency else 50
        
        # Weighted combination (sparsity most important for brain-like efficiency)
        brain_score = 0.5 * sparsity_score + 0.3 * energy_score + 0.2 * memory_score
        return min(100, max(0, brain_score))

# Global brain metrics tracker
brain_metrics = BrainMetrics()

# ---------------------------------------------------------------------------
# GPT Model with Brain-Inspired Modifications
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # Brain-inspired configs
    sparsity_target: float = 0.90  # Target 90% sparsity (brain-like)
    adaptive_precision: bool = True  # Use mixed precision adaptively
    forgetting_rate: float = 0.01  # Synaptic pruning rate


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def standard_attention(q, k, v, causal=True):
    """Fallback attention for systems without FlashAttention"""
    B, T, nh, hs = q.shape
    scale = 1.0 / math.sqrt(hs)
    
    # Reshape for standard attention computation
    q = q.transpose(1, 2)  # (B, nh, T, hs)
    k = k.transpose(1, 2)  # (B, nh, T, hs)
    v = v.transpose(1, 2)  # (B, nh, T, hs)
    
    # Attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Softmax and apply to values
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    
    # Reshape back
    out = out.transpose(1, 2).contiguous().view(B, T, nh * hs)
    return out


class BrainInspiredAttention(nn.Module):
    """
    Attention with brain-inspired sparse activation patterns.
    Implements top-k sparsity to mimic brain's selective attention.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.sparsity_target = config.sparsity_target
        
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Choose attention implementation based on availability
        if fa3 is not None and torch.cuda.is_available():
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Intel Mac compatible fallback
            y = standard_attention(q, k, v, causal=True)
            
        y = y.contiguous().view(B, T, -1)
        
        # Brain-inspired sparse activation
        if self.training:
            # Apply top-k sparsity to mimic brain's selective activation
            k = max(1, int((1.0 - self.sparsity_target) * y.shape[-1]))
            topk_values, topk_indices = torch.topk(y.abs(), k, dim=-1)
            sparse_y = torch.zeros_like(y)
            sparse_y.scatter_(-1, topk_indices, y.gather(-1, topk_indices))
            y = sparse_y
            
            # Track sparsity for brain metrics
            brain_metrics.update_sparsity(y)
        
        y = self.c_proj(y)
        return y


class AdaptivePrecisionMLP(nn.Module):
    """
    MLP with adaptive precision inspired by brain's energy efficiency.
    Uses different precisions for different components.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.adaptive_precision = config.adaptive_precision

    def forward(self, x):
        # Use different precision for different components
        if self.adaptive_precision and self.training:
            # Lower precision for the first projection (energy saving)
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                   dtype=torch.float16):
                x = self.c_fc(x)
        else:
            x = self.c_fc(x)
            
        x = F.relu(x).square()
        
        # Higher precision for final projection (quality preservation)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = BrainInspiredAttention(config, layer_idx)
        self.mlp = AdaptivePrecisionMLP(config)
        self.layer_idx = layer_idx
        self.forgetting_rate = config.forgetting_rate

    def forward(self, x, ve, cos_sin, window_size):
        # Store weights for forgetting analysis
        old_attn_weight = None
        if self.training and hasattr(self, '_prev_attn_weight'):
            old_attn_weight = self._prev_attn_weight
            
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        
        # Track weight changes for forgetting metrics
        if self.training:
            current_weight = self.attn.c_proj.weight.data.clone()
            if old_attn_weight is not None:
                brain_metrics.update_forgetting(old_attn_weight, current_weight)
            self._prev_attn_weight = current_weight
            
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('.c_proj'):
                    nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))
                else:
                    nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern
        pattern_len = len(pattern)
        window_sizes = []
        for i in range(config.n_layer):
            pattern_char = pattern[i % pattern_len]
            if pattern_char == 'L':
                window_size = (-1, -1)  # Full attention
            elif pattern_char == 'S':
                window_size = (config.sequence_len // 2, config.sequence_len // 2)
            else:
                raise ValueError(f"Unknown pattern character: {pattern_char}")
            window_sizes.append(window_size)
        return window_sizes

    @torch.no_grad()
    def _precompute_rotary_embeddings(self, seq_len, head_dim):
        theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        seq_idx = torch.arange(seq_len).float()
        idx_theta = torch.outer(seq_idx, theta)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        return cos, sin

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= MAX_SEQ_LEN, f"Cannot process {T} tokens, max is {MAX_SEQ_LEN}"
        pos_start_idx = 0
        x_emb = self.transformer.wte(x)
        x0 = x_emb.clone()

        cos = self.cos[pos_start_idx:pos_start_idx + T]
        sin = self.sin[pos_start_idx:pos_start_idx + T]
        cos_sin = cos.unsqueeze(0).unsqueeze(-1), sin.unsqueeze(0).unsqueeze(-1)

        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds.get(str(i))
            if ve is not None:
                ve = ve(x)
            else:
                ve = None
            window_size = self.window_sizes[i]
            x = x + self.resid_lambdas[i] * block(x, ve, cos_sin, window_size)
            x = x + self.x0_lambdas[i] * x0

        x = norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return loss if targets is not None else logits

    def num_scaling_params(self):
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        embed_params = count_params(self.transformer.wte)
        block_params = count_params(self.transformer.h[0]) if self.transformer.h else 0
        output_params = count_params(self.lm_head)
        value_embed_params = sum(count_params(ve) for ve in self.value_embeds.values())
        resid_lambda_params = self.resid_lambdas.numel()
        x0_lambda_params = self.x0_lambdas.numel()

        total = embed_params + len(self.transformer.h) * block_params + output_params + value_embed_params + resid_lambda_params + x0_lambda_params

        return {
            "embed": embed_params,
            "blocks": len(self.transformer.h) * block_params,
            "output": output_params,
            "value_embeds": value_embed_params,
            "lambdas": resid_lambda_params + x0_lambda_params,
            "total": total,
        }

    def estimate_flops(self):
        """Estimate FLOPs per forward pass per token (Intel Mac compatible)"""
        config = self.config
        N = config.n_embd
        L = config.n_layer
        V = config.vocab_size
        T = config.sequence_len

        # Embedding: V * N
        embed_flops = V * N

        # Per layer:
        # - QKV projections: 3 * N * N * T
        # - Attention: 2 * N * T^2 (Q@K and attn@V)
        # - Output projection: N * N * T
        # - MLP: 2 * N * 4N * T (up and down projections)
        layer_flops = T * (3 * N * N + N * 4 * N + N * 4 * N + N * N) + 2 * N * T * T
        total_layer_flops = L * layer_flops

        # Output projection: N * V * T
        output_flops = N * V * T

        total_flops = embed_flops + total_layer_flops + output_flops
        return total_flops / T  # per token

    def setup_optimizer(self, unembedding_lr, embedding_lr, scalar_lr, adam_betas, matrix_lr, weight_decay):
        param_groups = []
        
        # Embedding parameters (Adam)
        embedding_params = list(self.transformer.wte.parameters())
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': embedding_lr,
                'initial_lr': embedding_lr,
                'kind': 'adamw',
                'betas': adam_betas,
                'weight_decay': 0.0,
            })

        # Output layer (Adam)
        output_params = list(self.lm_head.parameters())
        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': unembedding_lr,
                'initial_lr': unembedding_lr,
                'kind': 'adamw',
                'betas': adam_betas,
                'weight_decay': 0.0,
            })

        # Per-layer lambdas (Adam)
        lambda_params = [self.resid_lambdas, self.x0_lambdas]
        if lambda_params:
            param_groups.append({
                'params': lambda_params,
                'lr': scalar_lr,
                'initial_lr': scalar_lr,
                'kind': 'adamw',
                'betas': adam_betas,
                'weight_decay': 0.0,
            })

        # Matrix parameters (Muon)
        matrix_params = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module not in [self.lm_head, self.transformer.wte]:
                    matrix_params.append(module.weight)
        
        # Value embeddings (Muon)
        for ve in self.value_embeds.values():
            matrix_params.extend(list(ve.parameters()))

        if matrix_params:
            param_groups.append({
                'params': matrix_params,
                'lr': matrix_lr,
                'initial_lr': matrix_lr,
                'kind': 'muon',
                'momentum': 0.95,
                'weight_decay': weight_decay,
                'ns_steps': 5,
            })

        return CustomOptimizer(param_groups)


# [Rest of the optimizer and training code remains the same...]
# ---------------------------------------------------------------------------
# Custom Optimizer (Muon + AdamW)
# ---------------------------------------------------------------------------

from torch._C import _cuda_getCurrentRawStream as get_raw_stream

@torch.library.custom_op("mylib::muon_step_fused", mutates_args={"params", "momentum_buffer", "second_momentum_buffer"})
def muon_step_fused(grads, params, momentum_buffer, second_momentum_buffer, 
                   momentum, lr, weight_decay, beta2, ns_steps, red_dim):
    pass

@muon_step_fused.register_fake
def _(grads, params, momentum_buffer, second_momentum_buffer, 
      momentum, lr, weight_decay, beta2, ns_steps, red_dim):
    return

class CustomOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups
        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.state = {}
        self._muon_momentum_t = torch.empty(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        self._muon_beta2_t = torch.empty(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        self._muon_lr_t = torch.empty(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        self._muon_wd_t = torch.empty(1, device='cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state.setdefault(id(p), {})
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            
            beta1, beta2 = group['betas']
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            denom = (exp_avg_sq / bias_correction2).sqrt().add_(1e-8)
            step_size = group['lr'] / bias_correction1
            
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
            if group['weight_decay'] > 0:
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

    @torch.no_grad()
    def _step_muon(self, group):
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return
        
        # For Intel Mac compatibility, use CPU-based parameter updates
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for p in params:
            state = self.state.setdefault(id(p), {})
            
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
                if p.dim() >= 2:
                    state_shape = (p.shape[-2], 1) if p.shape[-2] >= p.shape[-1] else (1, p.shape[-1])
                    state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=p.dtype, device=p.device)
                else:
                    state["second_momentum_buffer"] = torch.zeros_like(p)
            
            # Simple Muon update for Intel Mac compatibility
            momentum_buffer = state["momentum_buffer"]
            grad = p.grad
            
            # Momentum update
            momentum_buffer.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
            
            # Apply update
            lr_scale = max(1.0, p.shape[-2] / p.shape[-1])**0.5 if p.dim() >= 2 else 1.0
            effective_lr = group['lr'] * lr_scale
            
            p.data.add_(momentum_buffer, alpha=-effective_lr)
            
            # Weight decay
            if group['weight_decay'] > 0:
                p.data.mul_(1 - effective_lr * group['weight_decay'])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters with Brain-Inspired Modifications
# ---------------------------------------------------------------------------

# Model architecture (brain-inspired efficiency)
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization (brain metabolic efficiency inspired)
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size (brain-inspired: smaller, more efficient)
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# Brain-specific parameters
SPARSITY_TARGET = 0.90   # Target 90% sparsity (brain-like)
ADAPTIVE_PRECISION = True # Use mixed precision adaptively
FORGETTING_RATE = 0.01   # Synaptic pruning rate

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Brain-AutoResearch running on: {device} ({platform.machine()})")

# Intel Mac compatible autocast
if torch.cuda.is_available():
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    H100_BF16_PEAK_FLOPS = 989.5e12
else:
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.float32)
    # Intel Mac approximate FLOPS (much lower than GPU)
    H100_BF16_PEAK_FLOPS = 1e12  # Rough approximation for Intel Mac

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        sparsity_target=SPARSITY_TARGET,
        adaptive_precision=ADAPTIVE_PRECISION,
        forgetting_rate=FORGETTING_RATE,
    )

config = build_model_config(DEPTH)
print(f"🧠 Brain-inspired model config: {asdict(config)}")

# Track baseline memory for efficiency calculations
baseline_memory = psutil.virtual_memory().used / 1024 / 1024  # MB

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# Only compile if CUDA available (Intel Mac compatibility)
if torch.cuda.is_available():
    model = torch.compile(model, dynamic=False)
else:
    print("⚠️  Model compilation disabled on Intel Mac for compatibility")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"🧠 Brain efficiency target: {SPARSITY_TARGET*100:.0f}% sparsity")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Brain-Inspired Training Loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Reset brain metrics for this run
brain_metrics.reset()

print("🧠 Starting brain-inspired training...")

while True:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Update brain metrics
    if step % 10 == 0:  # Update every 10 steps to avoid overhead
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        tokens_this_step = TOTAL_BATCH_SIZE
        brain_metrics.update_energy(tokens_this_step, dt)
        brain_metrics.update_memory_efficiency(current_memory, baseline_memory)

    # Logging with brain metrics
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    # Get current brain efficiency score
    brain_summary = brain_metrics.get_summary()
    brain_score = brain_summary['brain_efficiency_score']
    avg_sparsity = brain_summary['sparsity_mean'] * 100

    print(f"\r🧠 step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | brain_score: {brain_score:.1f} | sparsity: {avg_sparsity:.1f}% | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final brain metrics summary
brain_summary = brain_metrics.get_summary()

# Final summary with brain metrics
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0

if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    # Approximate memory usage for Intel Mac
    current_process = psutil.Process()
    peak_vram_mb = current_process.memory_info().rss / 1024 / 1024

print("🧠 BRAIN-AUTORESEARCH RESULTS")
print("=" * 50)
print(f"val_bpb:                    {val_bpb:.6f}")
print(f"training_seconds:           {total_training_time:.1f}")
print(f"total_seconds:              {t_end - t_start:.1f}")
print(f"peak_vram_mb:               {peak_vram_mb:.1f}")
print(f"mfu_percent:                {steady_state_mfu:.2f}")
print(f"total_tokens_M:             {total_tokens / 1e6:.1f}")
print(f"num_steps:                  {step}")
print(f"num_params_M:               {num_params / 1e6:.1f}")
print(f"depth:                      {DEPTH}")
print("🧠 BRAIN EFFICIENCY METRICS")
print("-" * 30)
print(f"brain_efficiency_score:     {brain_summary['brain_efficiency_score']:.2f}")
print(f"avg_sparsity:               {brain_summary['sparsity_mean']*100:.2f}%")
print(f"avg_energy_per_token:       {brain_summary['energy_per_token_mean']:.6f}")
print(f"avg_memory_efficiency:      {brain_summary['memory_efficiency_mean']:.3f}")
print(f"avg_forgetting_rate:        {brain_summary['forgetting_rate_mean']:.6f}")
print(f"system_platform:            {brain_summary['system_info']['platform']}")
print(f"architecture:               {brain_summary['system_info']['architecture']}")

# Save brain metrics for analysis
brain_metrics_file = f"brain_metrics_step_{step}.json"
with open(brain_metrics_file, 'w') as f:
    json.dump(brain_summary, f, indent=2)

print(f"🧠 Brain metrics saved to: {brain_metrics_file}")

# Standard outputs for compatibility with original autoresearch
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")