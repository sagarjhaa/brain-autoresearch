# 🧠 Brain-AutoResearch Quick Start

**Ready to discover brain-inspired AI efficiency patterns tonight!**

## ⚡ 5-Minute Setup

```bash
cd brain-autoresearch
./setup_brain_research.sh
```

This will:
1. Install dependencies (Intel Mac compatible)
2. Download training data
3. Run baseline experiment (5min)
4. Initialize brain metrics tracking

## 🧪 Start Experimenting

### Option 1: High Sparsity Research
Focus on achieving 90%+ activation sparsity like biological neurons:

```bash
python experiments/brain_presets.py
# Copy "sparse" preset to train.py hyperparameters section
uv run train.py
```

### Option 2: Energy Efficiency Research  
Minimize energy per token for edge deployment:

```bash
# Use "energy" preset from brain_presets.py
# Focus on adaptive precision patterns
```

### Option 3: Autonomous Discovery
Let the system run overnight and discover patterns:

```bash
# Start with baseline config
# System will autonomously try improvements
# Wake up to breakthrough results!
nohup uv run train.py > overnight_research.log 2>&1 &
```

## 📊 Monitor Progress

### Real-time Brain Metrics
```bash
tail -f run.log | grep "brain_score\|sparsity"
```

### Track All Results
```bash
cat results.tsv
```

### Example Output
```
🧠 step 00953 (100.0%) | loss: 0.997900 | brain_score: 78.4 | sparsity: 89.2% | ...
```

## 🎯 Success Targets

**Excellent Brain-Efficiency:**
- brain_efficiency_score > 85
- avg_sparsity > 90% 
- val_bpb < 1.0
- Compatible with Intel Mac ✅

## 🧠 Key Research Areas

### 1. Sparse Activation Patterns
- **Target**: 85-95% sparsity (brain-level)
- **Try**: Top-k vs threshold sparsity
- **Preset**: `sparse` or `extreme_sparse`

### 2. Energy Efficiency
- **Target**: Minimize energy per token
- **Try**: Adaptive precision, early exit
- **Preset**: `energy` or `low_power`

### 3. Forgetting Mechanisms  
- **Target**: Active information pruning
- **Try**: Weight decay scheduling, synaptic pruning
- **Preset**: `forgetting` or `sleep`

### 4. Hierarchical Processing
- **Target**: Brain-like layered processing
- **Try**: Progressive attention, cortical patterns  
- **Preset**: `hierarchical` or `cortex`

## 🔬 Experimental Presets

All presets are in `experiments/brain_presets.py`:

```python
# View all presets
python experiments/brain_presets.py

# Get specific preset instructions
# Then copy config to train.py hyperparameters
```

### Available Presets:
- **sparse**: High sparsity (92%)
- **extreme_sparse**: Extreme sparsity (96%)  
- **energy**: Energy-optimal architecture
- **low_power**: Intel Mac optimized
- **forgetting**: Active synaptic pruning
- **sleep**: Sleep-like consolidation
- **hierarchical**: Multi-scale processing
- **cortex**: Cortical layer patterns

## 📋 Results Tracking

Brain-specific metrics are logged to `results.tsv`:

```
commit	val_bpb	memory_gb	brain_score	sparsity	energy_per_token	status	description
a1b2c3d	0.997900	44.0	78.45	89.23	0.000123	keep	baseline brain config
b2c3d4e	0.993200	44.2	82.10	91.45	0.000098	keep	92% sparsity target
```

## 🛠️ Troubleshooting

### Intel Mac Issues
- ✅ System auto-detects Intel Mac 
- ✅ Disables CUDA optimizations
- ✅ Falls back to CPU-friendly implementations
- ✅ Uses standard attention when FlashAttention unavailable

### Memory Issues
- Reduce `DEVICE_BATCH_SIZE` in train.py
- Try `low_power` preset
- Use smaller `DEPTH` values

### Performance Issues
- Intel Mac: Expected ~10x slower than GPU
- Still valuable for brain-pattern research
- Focus on efficiency ratios, not absolute speed

## 🌍 Research Impact

Every efficiency pattern you discover could:
- Enable AI on $100 smartphones globally
- Reduce AI energy consumption 10x
- Democratize AI for developing nations  
- Pioneer sustainable artificial intelligence

## 🚀 Tonight's Mission

1. **Run setup** (5 minutes)
2. **Pick a research focus** (sparsity, energy, forgetting, or hierarchical)  
3. **Start experimenting** with brain presets
4. **Let it run overnight** for autonomous discovery
5. **Wake up to breakthrough insights!** 

The brain achieved intelligence under extreme efficiency constraints. Let's learn how! 🧠⚡

---

*"In the efficiency patterns of the brain, we find the path to truly scalable AI."*