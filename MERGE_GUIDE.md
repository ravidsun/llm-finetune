# Model Merging Guide

Complete guide for merging your trained LoRA adapter with the base model to create a standalone merged model.

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is Model Merging?](#what-is-model-merging)
3. [Requirements](#requirements)
4. [Usage Options](#usage-options)
5. [Testing the Merged Model](#testing-the-merged-model)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Automatic Merge (Recommended)

```bash
cd /workspace/llm-finetune
bash scripts/merge.sh
```

This will:
- ✅ Auto-detect your base model (7B or 14B)
- ✅ Find your trained adapter in `output/`
- ✅ Merge them into `merged_model/`
- ✅ Include tokenizer and all necessary files

### Option 2: Manual Merge with Python Script

```bash
cd /workspace/llm-finetune

python scripts/merge_model.py \
    --adapter_path /workspace/llm-finetune/output \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_path /workspace/llm-finetune/merged_model
```

---

## What is Model Merging?

### Before Merge (LoRA Adapter)
- **Two components needed:**
  - Base model: `Qwen/Qwen2.5-7B-Instruct` (downloaded from HuggingFace)
  - Adapter: `adapter_model.safetensors` (~200-500 MB)
- **Requires:** Loading both components at inference time
- **Pros:** Small file size, easy to share adapter only
- **Cons:** Slightly slower loading, requires PEFT library

### After Merge (Standalone Model)
- **Single component:**
  - Merged model: All weights in one place (~15-30 GB for 7B/14B)
- **Requires:** Just transformers library
- **Pros:** Faster loading, simpler deployment, standard HuggingFace format
- **Cons:** Larger file size

**When to merge:**
- ✅ Production deployment
- ✅ Sharing the full model
- ✅ Using with standard inference tools (vLLM, TGI, etc.)
- ✅ Simplifying deployment pipeline

**When NOT to merge:**
- ❌ Still experimenting with different adapters
- ❌ Want to keep file sizes small
- ❌ Need to switch between multiple adapters

---

## Requirements

### Disk Space
- **7B Model:** ~30 GB free space
- **14B Model:** ~60 GB free space
- **Why:** Merging requires space for base model + adapter + merged model

### Memory
- **CPU Merge:** Works on any system (slower)
- **GPU Merge:** Requires sufficient VRAM (faster)
  - 7B: ~20 GB VRAM
  - 14B: ~40 GB VRAM

### Dependencies
Already installed if you completed training:
```bash
pip install transformers peft torch accelerate safetensors
```

---

## Usage Options

### 1. Basic Merge (Auto-detect)

```bash
bash scripts/merge.sh
```

The script will automatically:
- Detect your base model from adapter config
- Use default paths
- Show progress and file sizes

### 2. Specify Base Model

```bash
# For 7B model
bash scripts/merge.sh --base 7B

# For 14B model
bash scripts/merge.sh --base 14B

# For custom model
bash scripts/merge.sh --base "custom/model-name"
```

### 3. Custom Paths

```bash
bash scripts/merge.sh \
    --adapter /workspace/llm-finetune/output \
    --output /workspace/my-merged-model
```

### 4. Push to HuggingFace Hub

```bash
# First, login to HuggingFace
huggingface-cli login --token YOUR_TOKEN

# Merge and push
bash scripts/merge.sh \
    --push \
    --hub-id username/my-finetuned-model
```

### 5. Using Python Script Directly

```bash
python scripts/merge_model.py \
    --adapter_path /workspace/llm-finetune/output \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_path /workspace/llm-finetune/merged_model \
    --device cuda \
    --max_shard_size 5GB
```

**Python Script Options:**
- `--adapter_path`: Path to your trained adapter
- `--base_model`: Base model name or path
- `--output_path`: Where to save merged model
- `--device`: `auto`, `cuda`, or `cpu` (default: `auto`)
- `--max_shard_size`: Max size per file (default: `5GB`)
- `--push_to_hub`: Push to HuggingFace Hub
- `--hub_model_id`: Hub model ID (required with `--push_to_hub`)

---

## Testing the Merged Model

### Quick Test

```bash
python scripts/test_merged_model.py /workspace/llm-finetune/merged_model
```

This will run 3 default test prompts and show responses.

### Custom Test Prompt

```bash
python scripts/test_merged_model.py \
    /workspace/llm-finetune/merged_model \
    --prompt "Your custom test prompt here"
```

### Advanced Testing

```bash
python scripts/test_merged_model.py \
    /workspace/llm-finetune/merged_model \
    --prompt "Explain quantum computing" \
    --max_length 1024 \
    --temperature 0.7 \
    --device cuda
```

**Test Script Options:**
- `--prompt`: Custom prompt to test
- `--max_length`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature 0.1-1.0 (default: 0.7)
- `--device`: `auto`, `cuda`, or `cpu`

### Manual Testing in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "/workspace/llm-finetune/merged_model",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/llm-finetune/merged_model"
)

# Test it
messages = [{"role": "user", "content": "Hello!"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Deployment

### 1. Download to Local Machine

```bash
# From your local machine (not RunPod)
scp -r root@POD_IP:/workspace/llm-finetune/merged_model ./my-model/
```

### 2. Upload to HuggingFace Hub

```bash
# Login first
huggingface-cli login --token YOUR_TOKEN

# Merge and push in one command
bash scripts/merge.sh --push --hub-id username/model-name
```

Or manually:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/workspace/llm-finetune/merged_model")
tokenizer = AutoTokenizer.from_pretrained("/workspace/llm-finetune/merged_model")

model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")
```

### 3. Use with Inference Servers

#### vLLM
```bash
pip install vllm

vllm serve /workspace/llm-finetune/merged_model \
    --host 0.0.0.0 \
    --port 8000
```

#### Text Generation Inference (TGI)
```bash
docker run -p 8080:80 \
    -v /workspace/llm-finetune/merged_model:/model \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /model
```

#### Ollama
```bash
# Create Modelfile
cat > Modelfile << EOF
FROM /workspace/llm-finetune/merged_model
PARAMETER temperature 0.7
EOF

# Create Ollama model
ollama create my-model -f Modelfile
ollama run my-model
```

### 4. Quantize for Smaller Size

```python
# GGUF format (for llama.cpp, Ollama)
# Install: pip install llama-cpp-python

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/llm-finetune/merged_model"
)

# Save in GGUF format (Q4_K_M quantization)
model.save_pretrained(
    "/workspace/llm-finetune/merged_model_gguf",
    safe_serialization=True,
    max_shard_size="5GB"
)
```

---

## Troubleshooting

### Issue: "Out of Disk Space"

**Solution:**
```bash
# Check available space
df -h /workspace

# Free up space before merging
rm -rf ~/.cache/huggingface/hub/models--*  # Remove cached models
rm -rf /workspace/llm-finetune/output/checkpoint-*  # Remove old checkpoints
pip cache purge
```

### Issue: "Out of Memory" during merge

**Solution 1:** Use CPU instead of GPU
```bash
python scripts/merge_model.py \
    --adapter_path /workspace/llm-finetune/output \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_path /workspace/llm-finetune/merged_model \
    --device cpu
```

**Solution 2:** Use lower precision
```python
# Edit merge_model.py, change torch_dtype:
torch_dtype=torch.float16  # or torch.bfloat16
```

### Issue: "Adapter not found"

**Verify adapter files exist:**
```bash
ls -lh /workspace/llm-finetune/output/

# Should see:
# - adapter_config.json
# - adapter_model.safetensors (or adapter_model.bin)
```

If missing, training may not have completed successfully.

### Issue: "Base model mismatch"

**Check which base model was used:**
```bash
# Read from adapter config
cat /workspace/llm-finetune/output/adapter_config.json | grep base_model

# Or check training config
cat /workspace/llm-finetune/config.yaml | grep model_name
```

Then specify correct base model:
```bash
bash scripts/merge.sh --base Qwen/Qwen2.5-7B-Instruct
```

### Issue: Merged model not generating properly

**Possible causes:**
1. **Wrong base model used** - Verify base model matches training
2. **Tokenizer mismatch** - Check tokenizer files are included
3. **Model not fully merged** - Re-run merge process

**Test with original adapter first:**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load with adapter (original way)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "/workspace/llm-finetune/output")

# If this works but merged doesn't, re-merge
```

---

## File Structure After Merge

```
merged_model/
├── config.json                    # Model configuration
├── generation_config.json         # Generation settings
├── model-00001-of-00003.safetensors  # Model weights (sharded)
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json   # Shard index
├── special_tokens_map.json        # Special tokens
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json          # Tokenizer config
├── vocab.json                     # Vocabulary
└── merges.txt                     # BPE merges
```

**Total size:**
- 7B model: ~15-20 GB
- 14B model: ~30-35 GB

---

## Best Practices

1. **Before Merging:**
   - ✅ Verify training completed successfully
   - ✅ Test adapter works correctly
   - ✅ Check available disk space (30-60 GB)
   - ✅ Backup adapter files

2. **During Merging:**
   - ✅ Monitor disk space: `watch -n 5 df -h /workspace`
   - ✅ Let process complete (can take 10-30 minutes)
   - ✅ Don't interrupt the process

3. **After Merging:**
   - ✅ Test merged model works correctly
   - ✅ Compare outputs with adapter version
   - ✅ Backup or upload merged model
   - ✅ Clean up temporary files

4. **Storage Management:**
   - Keep adapter (small, easy to share/experiment)
   - Merge for deployment only
   - Upload to Hub or download locally
   - Clean up RunPod instance after download

---

## Performance Comparison

### Loading Time

| Method | 7B Model | 14B Model |
|--------|----------|-----------|
| Adapter (PEFT) | ~30-45s | ~60-90s |
| Merged Model | ~20-30s | ~40-60s |

### Inference Speed

Both methods have similar inference speed once loaded.

### Compatibility

| Use Case | Adapter | Merged |
|----------|---------|--------|
| transformers | ✅ | ✅ |
| PEFT | ✅ | ❌ |
| vLLM | ⚠️ Limited | ✅ |
| TGI | ⚠️ Limited | ✅ |
| Ollama | ❌ | ✅ |
| llama.cpp | ❌ | ✅ |

---

## Cost Estimation

### RunPod Time
- **Merge process:** 10-30 minutes
- **A40 GPU:** ~$0.07-$0.20
- **CPU-only:** Free (longer time)

### Storage
- **RunPod Storage:** Included in pod
- **HuggingFace Hub:** Free (public repos)
- **Download Bandwidth:** Varies by provider

---

## Need Help?

- **Issue Tracker**: [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- **Documentation**: [Full Docs](docs/README.md)
- **Cleanup Guide**: [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)
- **RunPod Guide**: [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md)

---

## Quick Reference

```bash
# Merge (auto-detect)
bash scripts/merge.sh

# Merge 7B
bash scripts/merge.sh --base 7B

# Merge 14B
bash scripts/merge.sh --base 14B

# Test merged model
python scripts/test_merged_model.py /workspace/llm-finetune/merged_model

# Download to local
scp -r root@POD_IP:/workspace/llm-finetune/merged_model ./

# Upload to Hub
bash scripts/merge.sh --push --hub-id username/model-name

# Free up space after
rm -rf /workspace/llm-finetune/output/checkpoint-*
rm -rf ~/.cache/huggingface/
```
