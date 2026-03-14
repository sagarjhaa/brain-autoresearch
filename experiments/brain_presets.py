"""
Brain-Inspired Experimental Presets
Ready-to-run configurations for different brain efficiency research areas.

Usage: Copy parameters from these presets into train.py before running experiments.
"""

# =============================================================================
# 🧠 SPARSE ACTIVATION EXPERIMENTS
# =============================================================================

SPARSE_ACTIVATION_PRESET = {
    "name": "High Sparsity (Brain-Level)",
    "description": "Achieve 90%+ activation sparsity like biological neurons",
    "config": {
        "SPARSITY_TARGET": 0.92,  # 92% sparsity (brain-like)
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.005,  # Gentle forgetting
        "DEPTH": 6,  # Fewer layers, more efficient
        "ASPECT_RATIO": 96,  # Wider layers
        "DEVICE_BATCH_SIZE": 64,  # Smaller batches for stability
    },
    "expected_brain_score": "80-90",
    "research_focus": "sparse activation patterns, top-k vs threshold sparsity"
}

EXTREME_SPARSE_PRESET = {
    "name": "Extreme Sparsity (95%+)",
    "description": "Push sparsity to biological limits",
    "config": {
        "SPARSITY_TARGET": 0.96,  # 96% sparsity (extreme)
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.001,  # Very gentle to maintain information
        "DEPTH": 4,  # Even fewer layers
        "ASPECT_RATIO": 128,  # Very wide layers
        "DEVICE_BATCH_SIZE": 32,  # Small batches for numerical stability
        "MATRIX_LR": 0.02,  # Lower LR for stability
    },
    "expected_brain_score": "85-95",
    "research_focus": "stability at extreme sparsity levels"
}

# =============================================================================
# ⚡ ENERGY EFFICIENCY EXPERIMENTS  
# =============================================================================

ENERGY_EFFICIENCY_PRESET = {
    "name": "Energy-Optimal Architecture",
    "description": "Minimize energy per token while maintaining performance",
    "config": {
        "SPARSITY_TARGET": 0.88,  # Moderate sparsity for stability
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.01,
        "DEPTH": 6,  # Energy-efficient depth
        "ASPECT_RATIO": 64,  # Balanced width
        "DEVICE_BATCH_SIZE": 256,  # Larger batches for throughput
        "WINDOW_PATTERN": "SSSS",  # All sliding window (energy efficient)
        "MATRIX_LR": 0.06,  # Higher LR for faster convergence
    },
    "expected_brain_score": "75-85",
    "research_focus": "energy per token, adaptive precision patterns"
}

LOW_POWER_PRESET = {
    "name": "Ultra Low Power (Intel Mac)",
    "description": "Optimized for battery-powered devices",
    "config": {
        "SPARSITY_TARGET": 0.85,  # Conservative for CPU
        "ADAPTIVE_PRECISION": False,  # Simpler for CPU
        "FORGETTING_RATE": 0.02,  # More aggressive pruning
        "DEPTH": 4,  # Very shallow for speed
        "ASPECT_RATIO": 48,  # Narrow for memory efficiency  
        "DEVICE_BATCH_SIZE": 16,  # Small for memory
        "TOTAL_BATCH_SIZE": 2**16,  # Smaller total batch
    },
    "expected_brain_score": "70-80",
    "research_focus": "Intel Mac optimization, memory efficiency"
}

# =============================================================================
# 🧘 FORGETTING MECHANISM EXPERIMENTS
# =============================================================================

ACTIVE_FORGETTING_PRESET = {
    "name": "Active Forgetting (Synaptic Pruning)",
    "description": "Implement brain-like synaptic pruning mechanisms",
    "config": {
        "SPARSITY_TARGET": 0.90,
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.05,  # Aggressive forgetting
        "DEPTH": 8,  # Standard depth
        "ASPECT_RATIO": 64,
        "DEVICE_BATCH_SIZE": 128,
        "WEIGHT_DECAY": 0.4,  # Higher weight decay for pruning
        "WARMDOWN_RATIO": 0.7,  # Longer warmdown for consolidation
    },
    "expected_brain_score": "75-85",
    "research_focus": "forgetting rate optimization, weight pruning patterns"
}

SLEEP_CONSOLIDATION_PRESET = {
    "name": "Sleep-Like Consolidation",
    "description": "Mimic brain's sleep-based memory consolidation",
    "config": {
        "SPARSITY_TARGET": 0.87,
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.03,  # Moderate forgetting
        "DEPTH": 10,  # Deeper for complex patterns
        "ASPECT_RATIO": 64,
        "DEVICE_BATCH_SIZE": 64,  # Smaller batches for "sleep" phases
        "WARMDOWN_RATIO": 0.8,  # Very long consolidation phase
        "FINAL_LR_FRAC": 0.1,  # Don't go to zero (light sleep)
    },
    "expected_brain_score": "80-90",
    "research_focus": "consolidation phases, learning rate scheduling"
}

# =============================================================================
# 🏗️ HIERARCHICAL ARCHITECTURE EXPERIMENTS
# =============================================================================

HIERARCHICAL_PRESET = {
    "name": "Hierarchical Processing",
    "description": "Brain-like hierarchical information processing",
    "config": {
        "SPARSITY_TARGET": 0.89,
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.01,
        "DEPTH": 12,  # Deep hierarchy
        "ASPECT_RATIO": 48,  # Narrower per layer
        "DEVICE_BATCH_SIZE": 64,
        "WINDOW_PATTERN": "SSSSLLL",  # Progressive attention
    },
    "expected_brain_score": "75-85",
    "research_focus": "hierarchical attention patterns, progressive complexity"
}

CORTEX_INSPIRED_PRESET = {
    "name": "Cortex-Inspired Architecture",
    "description": "Mimic cortical processing patterns",
    "config": {
        "SPARSITY_TARGET": 0.94,  # Very sparse like cortex
        "ADAPTIVE_PRECISION": True,
        "FORGETTING_RATE": 0.008,
        "DEPTH": 16,  # Cortical depth
        "ASPECT_RATIO": 32,  # Many narrow layers
        "DEVICE_BATCH_SIZE": 32,
        "WINDOW_PATTERN": "SSSSSSSSLLLLLLLL",  # Local → global
    },
    "expected_brain_score": "85-95",
    "research_focus": "cortical layer patterns, local to global processing"
}

# =============================================================================
# 🎯 QUICK SETUP FUNCTIONS
# =============================================================================

def apply_preset_to_config(preset_config):
    """
    Generate the configuration code to paste into train.py
    """
    config_lines = []
    config_lines.append("# Brain-Inspired Preset Configuration")
    config_lines.append("# " + "="*50)
    
    for param, value in preset_config.items():
        if isinstance(value, str):
            config_lines.append(f'{param} = "{value}"')
        else:
            config_lines.append(f'{param} = {value}')
    
    return "\n".join(config_lines)

def get_preset_instructions(preset_name):
    """Get step-by-step instructions for applying a preset"""
    presets = {
        "sparse": SPARSE_ACTIVATION_PRESET,
        "extreme_sparse": EXTREME_SPARSE_PRESET,
        "energy": ENERGY_EFFICIENCY_PRESET,
        "low_power": LOW_POWER_PRESET,
        "forgetting": ACTIVE_FORGETTING_PRESET,
        "sleep": SLEEP_CONSOLIDATION_PRESET,
        "hierarchical": HIERARCHICAL_PRESET,
        "cortex": CORTEX_INSPIRED_PRESET,
    }
    
    if preset_name not in presets:
        return f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
    
    preset = presets[preset_name]
    
    instructions = f"""
🧠 {preset['name']} Experiment Setup

Description: {preset['description']}
Expected Brain Score: {preset['expected_brain_score']}
Research Focus: {preset['research_focus']}

Steps:
1. Open brain-autoresearch/train.py
2. Find the "Hyperparameters" section (around line 400)
3. Replace the parameter values with these:

{apply_preset_to_config(preset['config'])}

4. Save and run: uv run train.py
5. Monitor brain metrics: grep "brain_score\|sparsity" run.log

Success Indicators:
- brain_efficiency_score > {preset['expected_brain_score'].split('-')[0]}
- Training completes without crashes
- Sparsity levels match target
- Compatible with Intel Mac
"""
    
    return instructions

if __name__ == "__main__":
    # Print all available presets
    print("🧠 Brain-AutoResearch Experimental Presets")
    print("="*60)
    
    presets = [
        ("sparse", "High Sparsity"),
        ("extreme_sparse", "Extreme Sparsity"), 
        ("energy", "Energy Efficiency"),
        ("low_power", "Low Power"),
        ("forgetting", "Active Forgetting"),
        ("sleep", "Sleep Consolidation"),
        ("hierarchical", "Hierarchical"),
        ("cortex", "Cortex-Inspired"),
    ]
    
    for preset_id, preset_name in presets:
        print(f"\n{preset_name}:")
        print(get_preset_instructions(preset_id))