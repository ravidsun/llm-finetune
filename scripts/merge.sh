#!/bin/bash
# Merge LoRA Adapter with Base Model - RunPod Helper Script
#
# This script automatically detects your trained model and merges it
# with the base model to create a standalone merged model.
#
# Usage:
#   bash scripts/merge.sh                    # Auto-detect everything
#   bash scripts/merge.sh --base 7B          # Specify 7B base model
#   bash scripts/merge.sh --base 14B         # Specify 14B base model
#   bash scripts/merge.sh --push             # Push to HuggingFace Hub

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default paths
ADAPTER_PATH="/workspace/llm-finetune/output"
OUTPUT_PATH="/workspace/llm-finetune/merged_model"
PUSH_TO_HUB=false
HUB_MODEL_ID=""

# Parse arguments
BASE_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --base)
            if [[ "$2" == "7B" || "$2" == "7b" ]]; then
                BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
            elif [[ "$2" == "14B" || "$2" == "14b" ]]; then
                BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
            else
                BASE_MODEL="$2"
            fi
            shift 2
            ;;
        --adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --push)
            PUSH_TO_HUB=true
            shift
            ;;
        --hub-id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/merge.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --base MODEL      Base model to use (7B, 14B, or full model name)"
            echo "  --adapter PATH    Path to adapter (default: /workspace/llm-finetune/output)"
            echo "  --output PATH     Output path (default: /workspace/llm-finetune/merged_model)"
            echo "  --push            Push to HuggingFace Hub"
            echo "  --hub-id ID       HuggingFace Hub model ID (required with --push)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash scripts/merge.sh"
            echo "  bash scripts/merge.sh --base 7B"
            echo "  bash scripts/merge.sh --base 14B --output /workspace/my_model"
            echo "  bash scripts/merge.sh --push --hub-id username/my-model"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "🔄 Merge LoRA Adapter with Base Model"
echo "=================================================="
echo ""

# Step 1: Check if we're in RunPod environment
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}⚠️  Not in RunPod environment, using current directory${NC}"
    ADAPTER_PATH="./output"
    OUTPUT_PATH="./merged_model"
fi

# Step 2: Verify adapter exists
echo -e "${BLUE}[1/4] Checking adapter...${NC}"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo -e "${RED}❌ Adapter path does not exist: $ADAPTER_PATH${NC}"
    exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo -e "${RED}❌ No adapter_config.json found in $ADAPTER_PATH${NC}"
    echo -e "${RED}   Make sure training completed successfully${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Adapter found at: $ADAPTER_PATH${NC}"
echo ""

# Step 3: Auto-detect base model if not specified
if [ -z "$BASE_MODEL" ]; then
    echo -e "${BLUE}[2/4] Auto-detecting base model...${NC}"

    # Try to read from adapter config
    if command -v python3 &> /dev/null; then
        BASE_MODEL=$(python3 -c "import json; print(json.load(open('$ADAPTER_PATH/adapter_config.json'))['base_model_name_or_path'])" 2>/dev/null || echo "")
    fi

    # Fallback: check training config
    if [ -z "$BASE_MODEL" ] && [ -f "/workspace/llm-finetune/config.yaml" ]; then
        BASE_MODEL=$(grep "model_name:" /workspace/llm-finetune/config.yaml | awk '{print $2}' | tr -d '"' || echo "")
    fi

    # Final fallback: ask user
    if [ -z "$BASE_MODEL" ]; then
        echo -e "${YELLOW}Could not auto-detect base model.${NC}"
        echo ""
        echo "Which base model did you use for training?"
        echo "  1) Qwen/Qwen2.5-7B-Instruct"
        echo "  2) Qwen/Qwen2.5-14B-Instruct"
        echo "  3) Other (enter manually)"
        echo ""
        read -p "Enter choice (1-3): " choice

        case $choice in
            1)
                BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
                ;;
            2)
                BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
                ;;
            3)
                read -p "Enter base model name: " BASE_MODEL
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
    fi
fi

echo -e "${GREEN}✅ Base model: $BASE_MODEL${NC}"
echo ""

# Step 4: Check disk space
echo -e "${BLUE}[3/4] Checking disk space...${NC}"
AVAILABLE_SPACE=$(df /workspace 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

echo "   Available space: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 30 ]; then
    echo -e "${YELLOW}⚠️  Low disk space! Merging requires ~30-60 GB${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 5: Run merge script
echo -e "${BLUE}[4/4] Starting merge process...${NC}"
echo ""

# Build merge command
MERGE_CMD="python scripts/merge_model.py \
    --adapter_path $ADAPTER_PATH \
    --base_model $BASE_MODEL \
    --output_path $OUTPUT_PATH"

if [ "$PUSH_TO_HUB" = true ]; then
    if [ -z "$HUB_MODEL_ID" ]; then
        echo -e "${RED}❌ --hub-id required when using --push${NC}"
        exit 1
    fi
    MERGE_CMD="$MERGE_CMD --push_to_hub --hub_model_id $HUB_MODEL_ID"
fi

# Execute merge
$MERGE_CMD

# Done
echo ""
echo "=================================================="
echo -e "${GREEN}✅ MERGE COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "📁 Merged model saved to:"
echo "   $OUTPUT_PATH"
echo ""
echo "🧪 Test your merged model:"
echo "   python scripts/test_merged_model.py $OUTPUT_PATH"
echo ""
echo "📤 Download to local machine:"
echo "   scp -r root@POD_IP:$OUTPUT_PATH ./my-merged-model/"
echo ""
