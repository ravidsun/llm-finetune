#!/bin/bash
# Quick Setup Script for RunPod with LangChain
# Run this script to install all dependencies

set -e

echo "=========================================="
echo "  Installing Dependencies (LangChain Edition)"
echo "=========================================="

# Upgrade pip
pip install --upgrade pip

# Install core ML libraries
echo "Installing core ML libraries..."
pip install --upgrade \
    torch \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.27.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    bitsandbytes>=0.43.0 \
    safetensors>=0.4.0

# Install Flash Attention
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || echo "Flash attention skipped"

# Install LangChain ecosystem
echo "Installing LangChain ecosystem..."
pip install --upgrade \
    langchain>=0.2.0 \
    langchain-core>=0.2.0 \
    langchain-community>=0.2.0 \
    langchain-text-splitters>=0.2.0

# Install optional LLM providers
echo "Installing LLM providers for QA generation..."
pip install --upgrade \
    langchain-anthropic>=0.1.0 \
    anthropic>=0.25.0 \
    || echo "Anthropic installation optional"

# Install document processing
echo "Installing document processing libraries..."
pip install --upgrade \
    pymupdf>=1.24.0 \
    pdfplumber>=0.10.0 \
    python-docx>=1.1.0

# Install CLI dependencies
echo "Installing CLI dependencies..."
pip install --upgrade \
    typer[all]>=0.9.0 \
    rich>=13.0.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0

# Install utilities
echo "Installing utilities..."
pip install --upgrade \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    tiktoken>=0.6.0 \
    einops>=0.7.0 \
    jinja2>=3.1.0

# Install project
echo "Installing project..."
cd /workspace/llm-finetune-langchain 2>/dev/null || cd /app
pip install -e .

# Verify installation
echo ""
echo "=========================================="
echo "  Verifying Installation"
echo "=========================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import trl
print(f'TRL: {trl.__version__}')

import langchain
print(f'LangChain: {langchain.__version__}')

import langchain_core
print(f'LangChain Core: {langchain_core.__version__}')

try:
    import langchain_anthropic
    print(f'LangChain Anthropic: Available')
except:
    print('LangChain Anthropic: Not installed (optional)')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except:
    print('Flash Attention: Not installed')

print('\\n✓ All dependencies installed successfully!')
"

echo ""
echo "Setup complete! You can now run:"
echo "  python -m finetune_project --help"
