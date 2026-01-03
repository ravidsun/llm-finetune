#!/bin/bash
# RunPod Start Script with LangChain Support
# Sets up environment and runs the fine-tuning pipeline

set -e

echo "=========================================="
echo "  LLM Fine-Tuning Pipeline with LangChain"
echo "=========================================="

# Configuration
CONFIG_FILE="${CONFIG_FILE:-/workspace/config.yaml}"
DATA_PATH="${DATA_PATH:-/workspace/data}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/output}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"
ENABLE_QA="${ENABLE_QA:-false}"
ENABLE_AUGMENTATION="${ENABLE_AUGMENTATION:-false}"

# Set up caches
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"

# Set up Hugging Face token if provided
if [ -n "$HF_TOKEN" ]; then
    echo "Setting up Hugging Face authentication..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# Set up WandB if provided
if [ -n "$WANDB_API_KEY" ]; then
    echo "Setting up Weights & Biases..."
    wandb login "$WANDB_API_KEY"
fi

# Check for Anthropic API key (for QA generation)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "Anthropic API key detected - QA generation available"
fi

# Create directories
mkdir -p "$DATA_PATH" "$OUTPUT_PATH" "$HF_HOME"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    echo "Creating default config..."
    python -m finetune_project init --output "$CONFIG_FILE"
fi

echo ""
echo "Configuration: $CONFIG_FILE"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "QA Generation: $ENABLE_QA"
echo "Augmentation: $ENABLE_AUGMENTATION"
echo ""

# Download data if URL provided
if [ -n "$DATA_URL" ]; then
    echo "Downloading data from: $DATA_URL"
    wget -O "$DATA_PATH/data.zip" "$DATA_URL"
    unzip -o "$DATA_PATH/data.zip" -d "$DATA_PATH"
    rm "$DATA_PATH/data.zip"
fi

# Build prepare-data command
PREPARE_CMD="python -m finetune_project prepare-data --config $CONFIG_FILE"
if [ "$ENABLE_QA" = "true" ]; then
    PREPARE_CMD="$PREPARE_CMD --enable-qa"
fi
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
    PREPARE_CMD="$PREPARE_CMD --augment"
fi

# Prepare data (unless skipped)
if [ "$SKIP_PREPARE" != "true" ]; then
    echo ""
    echo "=========================================="
    echo "  Step 1: Preparing Data with LangChain"
    echo "=========================================="
    eval $PREPARE_CMD
fi

# Run training
echo ""
echo "=========================================="
echo "  Step 2: Training Model"
echo "=========================================="
python -m finetune_project train --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
echo "Output saved to: $OUTPUT_PATH"
