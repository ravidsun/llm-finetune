#!/bin/bash
# Automated RunPod Setup Script
# This script sets up everything needed for LLM fine-tuning on RunPod

set -e  # Exit on error

echo "=================================================="
echo "🚀 RunPod LLM Fine-Tuning - Automated Setup"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check GPU
echo -e "${BLUE}[1/7] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✅ GPU Detected: $GPU_NAME${NC}"
    echo -e "${GREEN}✅ VRAM: $GPU_MEMORY${NC}"
else
    echo "❌ No GPU detected! This script requires a GPU."
    exit 1
fi

# Step 2: Update system
echo ""
echo -e "${BLUE}[2/7] Updating system packages...${NC}"
apt-get update -qq

# Step 3: Navigate to workspace
echo ""
echo -e "${BLUE}[3/7] Setting up workspace...${NC}"
cd /workspace || { echo "❌ Failed to access /workspace"; exit 1; }

# Step 4: Clone repository (if not exists)
echo ""
echo -e "${BLUE}[4/7] Cloning repository...${NC}"
if [ -d "llm-finetune" ]; then
    echo -e "${YELLOW}⚠️  Repository already exists, updating...${NC}"
    cd llm-finetune
    git pull
else
    git clone https://github.com/ravidsun/llm-finetune.git
    cd llm-finetune
fi

echo -e "${GREEN}✅ Repository ready at: /workspace/llm-finetune${NC}"

# Step 5: Install dependencies
echo ""
echo -e "${BLUE}[5/7] Installing dependencies (this may take 3-5 minutes)...${NC}"
pip install -q -e .

# Step 6: Create directory structure
echo ""
echo -e "${BLUE}[6/7] Creating directory structure...${NC}"
mkdir -p /workspace/llm-finetune/data/input
mkdir -p /workspace/llm-finetune/processed_data
mkdir -p /workspace/llm-finetune/output
mkdir -p /workspace/llm-finetune/configs

echo -e "${GREEN}✅ Directories created:${NC}"
echo "   📁 /workspace/llm-finetune/data/input     (place your JSONL files here)"
echo "   📁 /workspace/llm-finetune/processed_data (processed data)"
echo "   📁 /workspace/llm-finetune/output         (training output)"
echo "   📁 /workspace/llm-finetune/configs        (config files)"

# Step 7: Verify installation
echo ""
echo -e "${BLUE}[7/7] Verifying installation...${NC}"
if python -m finetune_project --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Installation verified!${NC}"
else
    echo "❌ Installation verification failed"
    exit 1
fi

# Final summary
echo ""
echo "=================================================="
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "📖 Next Steps:"
echo "   1. Set your HuggingFace token:"
echo "      export HF_TOKEN='your_token_here'"
echo "      huggingface-cli login --token \$HF_TOKEN"
echo ""
echo "   2. Upload your training data to:"
echo "      /workspace/llm-finetune/data/input/"
echo ""
echo "   3. Start training with:"
echo "      cd /workspace/llm-finetune"
echo "      bash scripts/runpod_quick_train.sh"
echo ""
echo "📚 Full Guide: /workspace/llm-finetune/RUNPOD_GUIDE.md"
echo ""
