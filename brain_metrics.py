"""
Brain-Inspired Efficiency Metrics for AutoResearch
Tracks biological efficiency patterns: sparsity, energy, forgetting, hierarchical processing
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BrainMetrics:
    """Tracks brain-inspired efficiency metrics during training"""
    sparse_activation_ratio: float = 0.0
    energy_per_token: float = 0.0
    memory_compression_ratio: float = 0.0
    forgetting_efficiency: float = 0.0
    hierarchical_efficiency: float = 0.0
    neuron_death_ratio: float = 0.0  # Unused neurons
    adaptive_precision_ratio: float = 0.0
    
class BrainEfficiencyTracker:
    """Tracks and analyzes brain-like efficiency patterns"""
    
    def __init__(self):
        self.start_time = None
        self.start_energy = None
        self.activation_history = []
        self.weight_sparsity_history = []
        self.memory_usage_history = []
        self.layer_utilization = defaultdict(list)
        self.dead_neurons = set()
        self.precision_usage = []
        
    def start_measurement(self):
        """Start measuring energy and time"""
        self.start_time = time.time()
        # Approximate energy measurement via CPU usage
        self.start_energy = psutil.cpu_percent()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def measure_sparse_activation(self, model, input_batch):
        """Measure how sparsely neurons activate (brain: 1-4% active)"""
        total_neurons = 0
        active_neurons = 0
        
        hooks = []
        
        def activation_hook(module, input, output):
            nonlocal total_neurons, active_neurons
            if isinstance(output, torch.Tensor):
                flat_output = output.flatten()
                total_neurons += flat_output.numel()
                # Count neurons with activation > threshold (brain-inspired)
                threshold = 0.01 * flat_output.abs().max()
                active_neurons += (flat_output.abs() > threshold).sum().item()
        
        # Register hooks on all linear layers
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                hooks.append(module.register_forward_hook(activation_hook))
        
        with torch.no_grad():
            model(input_batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        sparsity_ratio = active_neurons / max(total_neurons, 1)
        self.activation_history.append(sparsity_ratio)
        return sparsity_ratio
    
    def measure_weight_sparsity(self, model):
        """Measure structural sparsity in weights"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                flat_param = param.flatten()
                total_params += flat_param.numel()
                # Count near-zero weights (biological pruning)
                threshold = 0.001 * flat_param.abs().std()
                zero_params += (flat_param.abs() < threshold).sum().item()
        
        sparsity = zero_params / max(total_params, 1)
        self.weight_sparsity_history.append(sparsity)
        return sparsity
    
    def measure_energy_per_token(self, tokens_processed):
        """Estimate energy consumption per token (brain: ~0.3 calories/hour)"""
        if self.start_time is None:
            return 0.0
            
        elapsed_time = time.time() - self.start_time
        current_cpu = psutil.cpu_percent()
        
        # Rough energy estimation (CPU-based)
        energy_delta = abs(current_cpu - self.start_energy) * elapsed_time
        energy_per_token = energy_delta / max(tokens_processed, 1)
        
        return energy_per_token
    
    def measure_memory_compression(self):
        """Track memory efficiency (brain: hierarchical compression)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            compression = allocated / max(cached, 1)
        else:
            # CPU memory approximation
            process = psutil.Process()
            memory_info = process.memory_info()
            compression = memory_info.rss / max(memory_info.vms, 1)
        
        self.memory_usage_history.append(compression)
        return compression
    
    def measure_layer_utilization(self, model):
        """Track which layers are most/least used (brain: different regions)"""
        layer_importance = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Measure gradient magnitude as proxy for layer importance
                grad_norm = param.grad.norm().item()
                layer_importance[name] = grad_norm
                self.layer_utilization[name].append(grad_norm)
        
        # Calculate hierarchical efficiency
        if layer_importance:
            values = list(layer_importance.values())
            hierarchical_eff = np.std(values) / (np.mean(values) + 1e-8)
        else:
            hierarchical_eff = 0.0
            
        return hierarchical_eff
    
    def detect_dead_neurons(self, model):
        """Find neurons that never activate (brain: neuronal death is normal)"""
        dead_count = 0
        total_neurons = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                flat_param = param.flatten()
                total_neurons += flat_param.numel()
                
                # Find weights that are consistently near zero
                threshold = 0.0001 * flat_param.abs().std()
                dead_mask = flat_param.abs() < threshold
                dead_count += dead_mask.sum().item()
        
        death_ratio = dead_count / max(total_neurons, 1)
        return death_ratio
    
    def measure_adaptive_precision(self, model):
        """Track if model uses different precisions efficiently"""
        precision_diversity = 0.0
        total_layers = 0
        
        for param in model.parameters():
            if param.requires_grad:
                total_layers += 1
                # Measure numerical diversity as proxy for precision needs
                param_std = param.std().item()
                param_range = (param.max() - param.min()).item()
                if param_range > 0:
                    precision_need = param_std / param_range
                    precision_diversity += precision_need
        
        avg_precision_diversity = precision_diversity / max(total_layers, 1)
        self.precision_usage.append(avg_precision_diversity)
        return avg_precision_diversity
    
    def calculate_brain_metrics(self, model, input_batch, tokens_processed) -> BrainMetrics:
        """Calculate comprehensive brain-inspired efficiency metrics"""
        
        # Sparse activation (brain: 1-4% neurons active)
        sparse_ratio = self.measure_sparse_activation(model, input_batch)
        
        # Weight sparsity (brain: massive connectivity but sparse usage)
        weight_sparsity = self.measure_weight_sparsity(model)
        
        # Energy efficiency (brain: 20W total)
        energy_per_token = self.measure_energy_per_token(tokens_processed)
        
        # Memory compression (brain: hierarchical storage)
        memory_compression = self.measure_memory_compression()
        
        # Hierarchical processing (brain: different cortical layers)
        hierarchical_eff = self.measure_layer_utilization(model)
        
        # Neuronal death (brain: programmed cell death is healthy)
        death_ratio = self.detect_dead_neurons(model)
        
        # Adaptive precision (brain: variable precision across regions)
        precision_ratio = self.measure_adaptive_precision(model)
        
        return BrainMetrics(
            sparse_activation_ratio=sparse_ratio,
            energy_per_token=energy_per_token,
            memory_compression_ratio=memory_compression,
            forgetting_efficiency=weight_sparsity,  # Using weight sparsity as forgetting proxy
            hierarchical_efficiency=hierarchical_eff,
            neuron_death_ratio=death_ratio,
            adaptive_precision_ratio=precision_ratio
        )
    
    def get_brain_score(self, metrics: BrainMetrics) -> float:
        """Calculate overall brain-efficiency score"""
        
        # Brain targets (based on neuroscience literature)
        BRAIN_TARGETS = {
            'sparse_activation': 0.03,  # 3% activation
            'energy_efficiency': 1.0,   # Normalized energy target
            'memory_compression': 0.8,  # High compression
            'hierarchical_efficiency': 0.5,  # Moderate hierarchy
            'healthy_death_ratio': 0.1,  # 10% dead neurons is healthy
            'precision_diversity': 0.3   # Moderate precision diversity
        }
        
        # Calculate distance from brain-optimal values
        sparse_score = 1.0 - abs(metrics.sparse_activation_ratio - BRAIN_TARGETS['sparse_activation']) / BRAIN_TARGETS['sparse_activation']
        energy_score = max(0, 1.0 - metrics.energy_per_token / 10.0)  # Penalize high energy
        memory_score = metrics.memory_compression_ratio
        hierarchy_score = min(1.0, metrics.hierarchical_efficiency)
        death_score = 1.0 - abs(metrics.neuron_death_ratio - BRAIN_TARGETS['healthy_death_ratio']) / BRAIN_TARGETS['healthy_death_ratio']
        precision_score = min(1.0, metrics.adaptive_precision_ratio)
        
        # Weighted combination (emphasize energy and sparsity)
        brain_score = (
            0.3 * sparse_score +      # Sparsity is critical
            0.3 * energy_score +      # Energy efficiency is critical  
            0.2 * memory_score +      # Memory efficiency
            0.1 * hierarchy_score +   # Hierarchical processing
            0.05 * death_score +      # Healthy pruning
            0.05 * precision_score    # Adaptive precision
        )
        
        return max(0.0, min(1.0, brain_score))
    
    def log_brain_metrics(self, metrics: BrainMetrics, step: int):
        """Log brain metrics in a format compatible with autoresearch"""
        brain_score = self.get_brain_score(metrics)
        
        print(f"Step {step} | Brain Efficiency Score: {brain_score:.4f}")
        print(f"  Sparse Activation: {metrics.sparse_activation_ratio:.4f} (target: ~0.03)")
        print(f"  Energy/Token: {metrics.energy_per_token:.6f}")
        print(f"  Memory Compression: {metrics.memory_compression_ratio:.4f}")
        print(f"  Hierarchical Eff: {metrics.hierarchical_efficiency:.4f}")
        print(f"  Dead Neurons: {metrics.neuron_death_ratio:.4f}")
        print(f"  Precision Diversity: {metrics.adaptive_precision_ratio:.4f}")
        
        return brain_score

def apply_brain_inspired_optimizations(model):
    """Apply brain-inspired optimizations to model"""
    
    # 1. Sparse Initialization (brain: sparse connectivity from birth)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # Initialize with sparse connectivity pattern
            with torch.no_grad():
                mask = torch.rand_like(module.weight) > 0.1  # 90% sparsity
                module.weight *= mask.float()
    
    # 2. Add adaptive sparsity hooks (brain: dynamic pruning)
    def sparsity_hook(module, input, output):
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                # Gradually increase sparsity (brain development pattern)
                threshold = 0.01 * module.weight.abs().mean()
                mask = module.weight.abs() > threshold
                module.weight *= mask.float()
    
    # Register sparsity hooks
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(sparsity_hook)
    
    return model