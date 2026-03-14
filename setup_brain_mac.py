#!/usr/bin/env python3
"""
Brain-AutoResearch Setup for Intel Mac
Sets up brain-inspired autoresearch environment optimized for CPU training
"""

import os
import sys
import torch
import platform
import subprocess
from pathlib import Path

def check_intel_mac():
    """Check if running on Intel Mac"""
    is_mac = platform.system() == "Darwin"
    is_intel = "Intel" in platform.processor() or platform.machine() == "x86_64"
    
    print(f"System: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"Intel Mac: {is_mac and is_intel}")
    
    return is_mac and is_intel

def setup_torch_intel():
    """Setup PyTorch for Intel Mac optimization"""
    print("Setting up PyTorch for Intel Mac...")
    
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")
    
    # Optimize for CPU
    cpu_count = os.cpu_count()
    torch.set_num_threads(max(1, cpu_count - 2))
    
    # Intel optimizations
    os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_count - 2))
    os.environ["MKL_NUM_THREADS"] = str(max(1, cpu_count - 2))
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max(1, cpu_count - 2))
    
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Intel MKL enabled: {torch.backends.mkl.enabled}")

def create_simple_prepare():
    """Create simplified prepare.py for Intel Mac"""
    prepare_content = '''"""
Simplified prepare.py for Intel Mac Brain-AutoResearch
"""

import torch
import numpy as np
from typing import Iterator, Tuple

# Brain-optimized constants for Intel Mac
MAX_SEQ_LEN = 256
TIME_BUDGET = 300  # 5 minutes
VOCAB_SIZE = 4096

class Tokenizer:
    """Simple character-level tokenizer for testing"""
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        return [ord(c) % self.vocab_size for c in text[:MAX_SEQ_LEN]]
    
    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens if t > 0])

class DummyDataLoader:
    """Dummy data loader for testing brain metrics"""
    def __init__(self, batch_size, seq_len, vocab_size, device):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.count = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count > 100:  # Generate ~100 batches
            raise StopIteration
            
        # Generate random text-like data
        data = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = data.clone()
        
        self.count += 1
        return data.to(self.device), targets.to(self.device)

def make_dataloader(split, batch_size, device):
    """Create data loader for brain training"""
    print(f"Creating dummy dataloader for {split} split")
    return DummyDataLoader(batch_size, MAX_SEQ_LEN, VOCAB_SIZE, device)

def evaluate_bpb(model, data_loader, device):
    """Evaluate bits per byte (simplified)"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(data_loader):
            if i > 10:  # Quick evaluation
                break
                
            logits, loss = model(input_ids, targets)
            total_loss += loss.item()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / max(i + 1, 1)
    bpb = avg_loss / np.log(2)  # Convert to bits per byte
    
    model.train()
    return bpb

print("Brain-optimized prepare.py loaded for Intel Mac")
'''
    
    with open("prepare.py", "w") as f:
        f.write(prepare_content)
    
    print("Created simplified prepare.py for Intel Mac")

def test_brain_setup():
    """Test brain-AutoResearch setup"""
    print("\nTesting Brain-AutoResearch setup...")
    
    try:
        from brain_metrics import BrainEfficiencyTracker
        print("✓ Brain metrics module loaded")
        
        # Test brain tracker
        tracker = BrainEfficiencyTracker()
        tracker.start_measurement()
        print("✓ Brain efficiency tracker initialized")
        
        # Test dummy model
        import torch.nn as nn
        dummy_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        dummy_input = torch.randn(1, 10, 256)
        metrics = tracker.calculate_brain_metrics(dummy_model, dummy_input, 100)
        brain_score = tracker.get_brain_score(metrics)
        
        print(f"✓ Brain metrics calculation successful")
        print(f"✓ Brain efficiency score: {brain_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing brain setup: {e}")
        return False

def main():
    print("🧠 Brain-AutoResearch Intel Mac Setup")
    print("="*50)
    
    # Check system
    if not check_intel_mac():
        print("Warning: Not detected as Intel Mac, but continuing...")
    
    print()
    
    # Setup PyTorch
    setup_torch_intel()
    print()
    
    # Create simple prepare.py
    create_simple_prepare()
    print()
    
    # Test setup
    success = test_brain_setup()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Brain-AutoResearch setup complete!")
        print("\nNext steps:")
        print("1. python3 train_brain.py  # Test brain training")
        print("2. Launch your favorite AI agent")
        print("3. Point it to brain_program.md")
        print("4. Let it discover brain-inspired optimizations!")
        print("\nGoal: Optimize neural networks to be as efficient as the human brain!")
    else:
        print("❌ Setup encountered issues. Check dependencies and try again.")
    
    print("\n🧠 Ready to make AI as efficient as biology! ⚡")

if __name__ == "__main__":
    main()