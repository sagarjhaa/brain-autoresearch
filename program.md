# brain-autoresearch

This is an experiment to have the LLM do its own research on **brain-inspired efficiency patterns** for AI systems. Unlike traditional autoresearch that optimizes purely for validation loss, brain-autoresearch optimizes for biological efficiency patterns that could enable deployment to billions of devices.

## 🧠 Brain-Inspired Research Goals

The human brain operates on ~20W power (12W for thinking) with ~100 billion neurons, but only 1-4% are active at any time. Our goal is to discover AI architectures that mimic these efficiency patterns while maintaining intelligence.

### Core Research Targets

1. **Sparse Activation Patterns** - Find optimal sparsity levels (target: 85-95% like brain)
2. **Energy Efficiency** - Minimize energy per useful operation  
3. **Adaptive Precision** - Use just enough precision for each computation
4. **Forgetting Mechanisms** - Actively discard irrelevant information
5. **Hierarchical Memory** - Multi-scale information organization

## Setup

To set up a new brain-autoresearch experiment:

1. **Agree on a brain-focused run tag**: propose a tag based on the brain pattern you're exploring (e.g. `sparse-mar14`, `energy-mar14`, `forget-mar14`). The branch `brain-autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b brain-autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the BRAIN-INSPIRED file you modify. Model architecture, optimizer, training loop, plus brain metrics.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with the brain-specific header row (see format below). The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the brain-inspired experimentation.

## 🧠 Brain-Specific Experimentation

Each experiment runs on a single device (GPU or Intel Mac CPU) for a **fixed time budget of 5 minutes**. You launch it as: `uv run train.py`.

**What you CAN do in train.py:**
- Modify sparsity targets and patterns
- Adjust energy efficiency parameters
- Implement forgetting mechanisms  
- Change adaptive precision strategies
- Experiment with hierarchical memory patterns
- Modify model architecture for brain-like efficiency
- Adjust optimizer hyperparameters
- Change batch sizes and model sizes

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation and data loading.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function is the ground truth.

**Brain-Specific Goals (in priority order):**
1. **Maximize brain_efficiency_score** - This combines sparsity, energy, and memory efficiency
2. **Minimize val_bpb** - Still need good performance
3. **Achieve target sparsity** - Aim for 85-95% activation sparsity
4. **Optimize energy per token** - Lower is better
5. **Maintain Intel Mac compatibility** - Should run on both GPU and CPU

**Complexity criterion for brain patterns**: Brain solutions should be *elegantly simple*. The brain achieves efficiency through simple principles applied consistently, not complex hacks. A 0.001 val_bpb improvement from adding complex sparsity logic? Probably not worth it. A 0.001 improvement from a simple, brain-inspired architectural change? Definitely keep.

**The first run**: Your very first run should always establish the baseline with current brain-inspired defaults.

## 🧠 Output Format

The brain-autoresearch script prints two summaries:

**Brain Metrics Summary:**
```
🧠 BRAIN-AUTORESEARCH RESULTS
==================================================
val_bpb:                    0.997900
training_seconds:           300.1
total_seconds:              325.9
peak_vram_mb:               45060.2
mfu_percent:                39.80
total_tokens_M:             499.6
num_steps:                  953
num_params_M:               50.3
depth:                      8
🧠 BRAIN EFFICIENCY METRICS
------------------------------
brain_efficiency_score:     78.45
avg_sparsity:               89.23%
avg_energy_per_token:       0.000123
avg_memory_efficiency:      1.234
avg_forgetting_rate:        0.001234
system_platform:            Darwin
architecture:               x86_64
```

**Standard Compatibility Summary:**
```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract brain metrics with:
```bash
grep "brain_efficiency_score:" run.log
grep "avg_sparsity:" run.log  
grep "avg_energy_per_token:" run.log
```

## 🧠 Brain-Specific Results Logging

Log experiments to `results.tsv` with brain-focused columns:

```
commit	val_bpb	memory_gb	brain_score	sparsity	energy_per_token	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes  
4. brain_efficiency_score (0-100) — use 0.0 for crashes
5. average sparsity percentage (0-100) — use 0.0 for crashes
6. average energy per token — use 0.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short description focusing on the brain pattern tested

Example:

```
commit	val_bpb	memory_gb	brain_score	sparsity	energy_per_token	status	description
a1b2c3d	0.997900	44.0	78.45	89.23	0.000123	keep	baseline brain-inspired config
b2c3d4e	0.993200	44.2	82.10	91.45	0.000098	keep	increase sparsity target to 92%
c3d4e5f	1.005000	44.0	75.20	87.80	0.000145	discard	too aggressive forgetting rate
d4e5f6g	0.000000	0.0	0.00	0.00	0.000000	crash	extreme sparsity (99%) caused NaN
```

## 🧠 Brain-Inspired Experiment Loop

The experiment runs on a dedicated brain-research branch (e.g. `brain-autoresearch/sparse-mar14`).

LOOP FOREVER with brain focus:

1. **Analyze current brain metrics** from the last run's output
2. **Hypothesize a brain-inspired improvement**:
   - Sparsity patterns: "What if I try top-k sparsity instead of threshold?"
   - Energy efficiency: "Can I reduce precision in attention but keep it high in output?"
   - Forgetting: "Should I prune weights more aggressively during training?"
   - Architecture: "Would fewer, wider layers be more brain-like?"
3. **Implement the change** in `train.py` 
4. **git commit** with descriptive brain-focused message
5. **Run experiment**: `uv run train.py > run.log 2>&1`
6. **Extract brain metrics**: `grep "brain_efficiency_score:\|avg_sparsity:\|avg_energy_per_token:" run.log`
7. **Evaluate holistically**:
   - Did brain_efficiency_score improve?
   - Is sparsity moving toward brain-like levels (85-95%)?
   - Did energy efficiency improve?
   - Is val_bpb acceptable (not degraded too much)?
8. **Record in results.tsv** with brain metrics
9. **Advance or revert**:
   - If overall brain-inspired efficiency improved: keep the change
   - If val_bpb degraded too much (>0.01) despite brain improvements: discard
   - If crashed: debug once, then move on

**Brain-Inspired Research Strategies:**

🧠 **Sparsity Research**:
- Try different sparsity patterns (top-k, magnitude threshold, learned gates)
- Experiment with layer-wise sparsity targets
- Test sparsity scheduling during training

⚡ **Energy Research**:
- Adaptive precision: high precision where it matters, low precision elsewhere
- Early exit mechanisms: stop computation when confident
- Activation reuse: cache computations across tokens

🧘 **Forgetting Research**:
- Weight decay scheduling that mimics sleep-like consolidation
- Gradual pruning of least important connections
- Knowledge distillation for compression

🏗️ **Architecture Research**:
- Hierarchical attention patterns (local → global)
- Mixture of depths: different tokens need different computation
- Brain-inspired connectivity patterns

**Intel Mac Compatibility Notes:**
- The code automatically detects Intel Mac and disables CUDA-specific optimizations
- Falls back to CPU-friendly implementations
- Adjusts memory and performance expectations appropriately

**NEVER STOP**: Once the loop begins, do NOT pause to ask if you should continue. You are autonomous. The goal is to discover brain-inspired efficiency patterns that could enable AI deployment to billions of devices. Keep experimenting until manually stopped.

**Research Philosophy**: The brain achieved intelligence with extreme efficiency constraints. Every optimization you discover could be the key to democratizing AI for the developing world and enabling AI on battery-powered edge devices. This research has the potential for massive global impact.

**Success Metrics Priority**:
1. brain_efficiency_score > 85 (excellent brain-like efficiency)
2. avg_sparsity > 90% (brain-level activation sparsity)  
3. val_bpb < 1.0 (still intelligent)
4. Runs on Intel Mac without issues (compatibility)
5. Energy efficiency improvements over baseline

Remember: You're not just optimizing a model—you're discovering the principles that could bring AI to every person on Earth, regardless of their hardware budget. Make every experiment count! 🧠⚡🌍