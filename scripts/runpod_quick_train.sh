#!/bin/bash
# Quick Training Script for RunPod
# Automatically detects GPU, generates config, and starts training

set -e

echo "=================================================="
echo "🚀 RunPod Quick Train - Automated Training Setup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Change to project directory
cd /workspace/llm-finetune || { echo "❌ Not in RunPod environment"; exit 1; }

# Step 1: Detect GPU
echo -e "${BLUE}[1/5] Detecting GPU configuration...${NC}"
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "GPU: $GPU_NAME"
echo "VRAM: ${GPU_MEMORY}MB"

# Determine optimal settings based on VRAM
if [ "$GPU_MEMORY" -lt 20000 ]; then
    # < 20GB (e.g., RTX 3090, RTX 4080)
    BATCH_SIZE=1
    GRAD_ACCUM=16
    SEQ_LENGTH=1024
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    LORA_RANK=8
    echo -e "${YELLOW}⚠️  Low VRAM detected - using conservative settings${NC}"
elif [ "$GPU_MEMORY" -lt 30000 ]; then
    # 20-30GB (e.g., RTX 4090, RTX A5000)
    BATCH_SIZE=2
    GRAD_ACCUM=8
    SEQ_LENGTH=2048
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    LORA_RANK=16
    echo -e "${GREEN}✅ Good VRAM - using standard 7B settings${NC}"
elif [ "$GPU_MEMORY" -lt 50000 ]; then
    # 30-50GB (e.g., A40, A100 40GB)
    BATCH_SIZE=4
    GRAD_ACCUM=4
    SEQ_LENGTH=4096
    MODEL="Qwen/Qwen2.5-14B-Instruct"
    LORA_RANK=32
    echo -e "${GREEN}✅ High VRAM - using 14B model settings${NC}"
else
    # 50GB+ (e.g., A100 80GB)
    BATCH_SIZE=8
    GRAD_ACCUM=2
    SEQ_LENGTH=4096
    MODEL="Qwen/Qwen2.5-14B-Instruct"
    LORA_RANK=64
    echo -e "${GREEN}✅ Very high VRAM - using optimized 14B settings${NC}"
fi

# Step 2: Check for training data
echo ""
echo -e "${BLUE}[2/5] Checking for training data...${NC}"
JSONL_COUNT=$(find data/input -name "*.jsonl" 2>/dev/null | wc -l)

if [ "$JSONL_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No JSONL files found in data/input/${NC}"
    echo ""
    echo "Please upload your training data:"
    echo "  1. Use RunPod UI: Connect → Upload Files"
    echo "  2. Or SCP: scp train.jsonl root@<POD-IP>:/workspace/llm-finetune/data/input/"
    echo "  3. Or create sample: python scripts/create_sample_data.py"
    exit 1
fi

echo -e "${GREEN}✅ Found $JSONL_COUNT JSONL file(s)${NC}"
TOTAL_LINES=$(wc -l data/input/*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
echo "   Total training examples: $TOTAL_LINES"

# Step 3: Generate config
echo ""
echo -e "${BLUE}[3/5] Generating optimized config...${NC}"

cat > config.yaml << EOF
model:
  model_name: "$MODEL"
  lora_rank: $LORA_RANK
  lora_alpha: $((LORA_RANK * 2))
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  use_qlora: true

data:
  input_path: "data/input"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false
    qa_generation_enabled: false

  augmentation:
    enabled: false

training:
  num_epochs: 3
  per_device_train_batch_size: $BATCH_SIZE
  gradient_accumulation_steps: $GRAD_ACCUM

  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"

  max_seq_length: $SEQ_LENGTH

  gradient_checkpointing: true
  fp16: false
  bf16: true

  output_dir: "/workspace/llm-finetune/output"
  save_strategy: "epoch"
  save_total_limit: 2

  logging_steps: 10
  logging_dir: "/workspace/llm-finetune/output/logs"
  report_to: []

  evaluation_strategy: "no"
EOF

echo -e "${GREEN}✅ Config generated: config.yaml${NC}"
echo ""
echo "Configuration Summary:"
echo "  Model: $MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max Sequence Length: $SEQ_LENGTH"
echo "  LoRA Rank: $LORA_RANK"

# Step 4: Prepare data
echo ""
echo -e "${BLUE}[4/5] Preparing data...${NC}"
python -m finetune_project prepare-data --config config.yaml

# Step 5: Start training
echo ""
echo -e "${BLUE}[5/5] Starting training in tmux session...${NC}"

# Check if tmux session exists
if tmux has-session -t training 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Tmux session 'training' already exists${NC}"
    echo "Options:"
    echo "  1. Kill existing session: tmux kill-session -t training"
    echo "  2. Attach to existing: tmux attach -t training"
    exit 0
fi

# Create training script
cat > /tmp/train_script.sh << 'SCRIPT_EOF'
#!/bin/bash
cd /workspace/llm-finetune
echo "🚀 Starting training..."
echo "⏱️  Started at: $(date)"
echo ""

python -m finetune_project train --config config.yaml 2>&1 | tee output/training.log

echo ""
echo "✅ Training completed at: $(date)"
SCRIPT_EOF

chmod +x /tmp/train_script.sh

# Start tmux session
tmux new-session -d -s training '/tmp/train_script.sh'

echo ""
echo "=================================================="
echo -e "${GREEN}✅ TRAINING STARTED!${NC}"
echo "=================================================="
echo ""
echo "📊 Monitor Training:"
echo "   tmux attach -t training    # Attach to session"
echo "   Ctrl+B, D                  # Detach from session"
echo ""
echo "📝 View Logs:"
echo "   tail -f output/training.log"
echo ""
echo "💻 Check GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "⏹️  Stop Training:"
echo "   tmux kill-session -t training"
echo ""
echo "Estimated training time:"
if [ "$TOTAL_LINES" -lt 1000 ]; then
    echo "   ~30-60 minutes (small dataset)"
elif [ "$TOTAL_LINES" -lt 5000 ]; then
    echo "   ~1-3 hours (medium dataset)"
else
    echo "   ~3-8 hours (large dataset)"
fi
echo ""
