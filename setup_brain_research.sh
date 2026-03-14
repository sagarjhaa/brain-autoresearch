#!/bin/bash

# 🧠 Brain-AutoResearch Setup Script
# Gets you ready to start discovering brain-efficiency patterns tonight!

set -e  # Exit on any error

echo "🧠 Brain-AutoResearch Setup"
echo "=================================="

# Check if we're on Intel Mac
PLATFORM=$(uname -m)
OS=$(uname -s)
echo "🖥️  Platform: $OS on $PLATFORM"

if [[ "$OS" == "Darwin" && "$PLATFORM" == "x86_64" ]]; then
    echo "✅ Intel Mac detected - optimizing for CPU compatibility"
    INTEL_MAC=true
else
    echo "🚀 GPU platform detected - full acceleration available"  
    INTEL_MAC=false
fi

echo ""

# 1. Install dependencies
echo "📦 Installing brain-autoresearch dependencies..."
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync
echo "✅ Dependencies installed"

# 2. Check for data
echo ""
echo "📊 Checking for training data..."
DATA_DIR="$HOME/.cache/autoresearch"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "📥 Preparing training data (this may take a few minutes)..."
    uv run prepare.py
    echo "✅ Data preparation complete"
else
    echo "✅ Training data found at $DATA_DIR"
fi

# 3. Run baseline experiment
echo ""
echo "🧪 Running baseline brain-inspired experiment..."
echo "This will take ~5 minutes and establish your baseline metrics."
echo ""

# Initialize results file with brain-specific headers
cat > results.tsv << EOF
commit	val_bpb	memory_gb	brain_score	sparsity	energy_per_token	status	description
EOF

echo "📋 Results tracking initialized: results.tsv"

# Run the baseline
echo "🚀 Starting baseline run..."
if uv run train.py > baseline_run.log 2>&1; then
    echo "✅ Baseline run completed successfully!"
    
    # Extract key metrics
    VAL_BPB=$(grep "^val_bpb:" baseline_run.log | cut -d: -f2 | tr -d ' ')
    BRAIN_SCORE=$(grep "brain_efficiency_score:" baseline_run.log | cut -d: -f2 | tr -d ' ')
    SPARSITY=$(grep "avg_sparsity:" baseline_run.log | sed 's/.*avg_sparsity:[[:space:]]*\([0-9.]*\)%.*/\1/')
    ENERGY=$(grep "avg_energy_per_token:" baseline_run.log | cut -d: -f2 | tr -d ' ')
    MEMORY=$(grep "peak_vram_mb:" baseline_run.log | cut -d: -f2 | tr -d ' ')
    
    # Convert memory to GB
    MEMORY_GB=$(echo "scale=1; $MEMORY / 1024" | bc -l)
    
    # Get commit hash
    COMMIT=$(git rev-parse --short HEAD)
    
    # Log to results
    echo -e "$COMMIT\t$VAL_BPB\t$MEMORY_GB\t$BRAIN_SCORE\t$SPARSITY\t$ENERGY\tkeep\tbaseline brain-inspired config" >> results.tsv
    
    echo ""
    echo "🧠 BASELINE BRAIN METRICS"
    echo "========================="
    echo "Validation BPB:        $VAL_BPB"
    echo "Brain Efficiency Score: $BRAIN_SCORE"
    echo "Average Sparsity:      $SPARSITY%"  
    echo "Energy per Token:      $ENERGY"
    echo "Memory Usage:          ${MEMORY_GB}GB"
    echo ""
    
    # Success criteria check
    BRAIN_SCORE_INT=$(echo "$BRAIN_SCORE" | cut -d. -f1)
    SPARSITY_INT=$(echo "$SPARSITY" | cut -d. -f1)
    
    if (( BRAIN_SCORE_INT >= 70 && SPARSITY_INT >= 85 )); then
        echo "🎯 Excellent! Your baseline already shows brain-like efficiency patterns!"
    elif (( BRAIN_SCORE_INT >= 50 && SPARSITY_INT >= 70 )); then
        echo "👍 Good baseline. Ready to optimize for brain efficiency!"
    else
        echo "📈 Baseline established. Plenty of room for brain-inspired improvements!"
    fi
    
else
    echo "❌ Baseline run failed. Check baseline_run.log for details."
    exit 1
fi

# 4. Show next steps
echo ""
echo "🎯 READY FOR BRAIN-EFFICIENCY RESEARCH!"
echo "======================================="
echo ""
echo "Your brain-autoresearch system is ready to discover efficiency patterns."
echo ""
echo "🧪 Quick Start Options:"
echo ""
echo "1. 🧠 High Sparsity Experiment (90%+ sparsity):"
echo "   python experiments/brain_presets.py"
echo "   # Copy 'sparse' preset config to train.py"
echo ""
echo "2. ⚡ Energy Efficiency Focus:"
echo "   # Focus on minimizing energy per token"
echo "   # Try adaptive precision and early exit"
echo ""
echo "3. 🧘 Forgetting Mechanisms:"
echo "   # Implement synaptic pruning patterns" 
echo "   # Test sleep-like consolidation"
echo ""
echo "4. 🚀 Autonomous Discovery:"
echo "   # Let the system run overnight"
echo "   # Wake up to efficiency breakthroughs!"
echo ""
echo "💡 Pro Tips:"
echo "- Monitor: tail -f run.log | grep 'brain_score\\|sparsity'"
echo "- Track progress: cat results.tsv"
echo "- Experiment presets: python experiments/brain_presets.py"
if [[ "$INTEL_MAC" == true ]]; then
    echo "- Intel Mac optimized: CPU-friendly implementations active ✅"
fi
echo ""
echo "🌍 Remember: Every efficiency pattern you discover could democratize AI"
echo "   for billions of people worldwide. Make every experiment count!"
echo ""
echo "🧠 Happy brain-inspired research! 🚀"