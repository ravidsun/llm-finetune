# Phase 1: Environment Setup

This guide covers the initial setup phase for the LLM fine-tuning pipeline.

## Prerequisites

- RunPod GPU Pod (RTX 4090 / A40 / A100) OR Local GPU machine
- Python 3.9+
- CUDA-compatible GPU with 16GB+ VRAM
- Git installed

## Step 1: Create RunPod Instance (Skip if using local machine)

1. Go to [RunPod.io](https://runpod.io) and login
2. Click "Deploy" → "GPU Pods"
3. Select GPU:
   - **7B models**: RTX 4090 (24GB) - $0.44/hr
   - **14B models**: A40 (46GB) - $0.39/hr
   - **32B models**: A100 (40GB) - $1.89/hr

4. Configure:
   - **Template**: RunPod PyTorch 2.x
   - **Container Disk**: 30GB
   - **Volume Disk**: 50GB mounted at `/workspace`

5. Click "Deploy On-Demand"
6. Wait for pod to start and note the SSH/HTTP connection details

## Step 2: Connect to Instance

### Via Web Terminal
Click "Connect" → "Start Web Terminal"

### Via SSH (Recommended)
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/runpod_key
```

## Step 3: Clone Repository

```bash
cd /workspace
git clone https://github.com/ravidsun/llm-finetune.git
cd llm-finetune
```

## Step 4: Set Environment Variables

Create a `.env` file with your API keys:

```bash
cat > /workspace/llm-finetune/.env <<'EOF'
# Required for gated models (Llama, Mistral)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: For QA generation from documents
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: For experiment tracking
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

Load environment variables:
```bash
source .env
export $(cat .env | xargs)
```

## Step 5: Install Dependencies

Run the setup script:

```bash
bash scripts/setup.sh
```

This installs:
- PyTorch with CUDA support
- Transformers, PEFT, TRL
- LangChain ecosystem
- Document processing libraries (PyMuPDF, pdfplumber)
- CLI tools (Typer, Rich)

**Installation time**: ~10-15 minutes

## Step 6: Verify Installation

Check that everything is installed correctly:

```bash
python -m finetune_project --help
```

You should see the CLI commands available.

Check GPU:
```bash
nvidia-smi
```

Verify libraries:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import transformers, peft, trl, langchain
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'LangChain: {langchain.__version__}')
"
```

## Step 7: Set Up Directories

Create working directories:

```bash
mkdir -p /workspace/data
mkdir -p /workspace/output
mkdir -p /workspace/hf_cache
mkdir -p /workspace/configs
```

## Troubleshooting

### Issue: CUDA not available
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of disk space
```bash
# Check disk usage
df -h

# Clean up unnecessary files
pip cache purge
rm -rf ~/.cache/huggingface/hub/*
```

### Issue: Permission denied
```bash
# Fix permissions
chmod +x scripts/*.sh
```

## Next Steps

✅ Setup complete! Proceed to [PHASE2_DATA_PREPARATION.md](PHASE2_DATA_PREPARATION.md)

## Quick Reference

```bash
# Workspace structure
/workspace/
├── llm-finetune/          # Project code
├── data/                  # Input data (PDFs, JSON, etc.)
├── output/                # Training outputs (LoRA adapters)
├── hf_cache/              # Hugging Face model cache
└── configs/               # Configuration files
```
