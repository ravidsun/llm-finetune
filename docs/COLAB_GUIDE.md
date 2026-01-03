# Google Colab Fine-Tuning Guide

Complete guide for fine-tuning LLMs using Google Colab's free/paid GPUs.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start (5 Cells)](#quick-start-5-cells)
- [Step-by-Step Setup](#step-by-step-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Saving Your Model](#saving-your-model)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is Google Colab?

Google Colab provides **free GPU access** (T4) and paid options (A100, V100) that work well for fine-tuning smaller models (7B) with LoRA/QLoRA.

### Colab GPU Options

| Tier | GPU | VRAM | Cost | Best For |
|------|-----|------|------|----------|
| **Free** | T4 | 15GB | Free | 7B models with QLoRA |
| **Colab Pro** | T4/V100 | 15-16GB | $10/month | 7B models, longer sessions |
| **Colab Pro+** | V100/A100 | 16-40GB | $50/month | 7B-13B models, priority access |

### Time & Cost Estimate

⏱️ **Total Time**: 45-90 minutes (including training)
💰 **Cost**: FREE (or $10-50/month for Pro/Pro+)
🎯 **Difficulty**: Beginner-friendly
📊 **Best For**: 7B models, small-medium datasets (1K-10K samples)

### Important Limitations

⚠️ **Colab Constraints**:
- **Session timeout**: 12 hours (free), 24 hours (Pro)
- **Idle timeout**: 90 minutes (free), longer (Pro)
- **Disk space**: ~80GB available
- **RAM**: 12-32GB depending on tier
- **No persistent storage** - Save to Drive or download models!

---

## Quick Start (5 Cells)

**Fastest way to get started!** Copy these 5 cells into a new Colab notebook.

### Cell 1: Complete Setup

```python
# ==========================================
# LLM Fine-Tuning - Complete Colab Setup
# ==========================================
# Run this cell first! It sets up everything you need.

print("🚀 Starting LLM Fine-Tuning Setup...")

# Step 1: Check GPU
print("\n📊 Step 1/7: Checking GPU...")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {vram_gb:.1f} GB")
else:
    print("❌ No GPU found!")
    print("Go to: Runtime → Change runtime type → Hardware accelerator: GPU")
    raise SystemExit

# Step 2: Install Dependencies
print("\n📦 Step 2/7: Installing dependencies (3-5 minutes)...")
!pip install -q -U torch torchvision torchaudio
!pip install -q -U transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.27.0
!pip install -q -U peft>=0.10.0 trl>=0.8.0 bitsandbytes>=0.43.0
!pip install -q -U langchain>=0.2.0 langchain-community>=0.2.0
!pip install -q -U pymupdf>=1.24.0 python-docx>=1.1.0
!pip install -q -U typer[all]>=0.9.0 rich>=13.0.0 pyyaml>=6.0
print("✅ Dependencies installed!")

# Step 3: Clone Repository
print("\n📂 Step 3/7: Cloning repository...")
import os
if os.path.exists('/content/llm-finetune'):
    print("⚠️ Repository already exists, updating...")
    %cd /content/llm-finetune
    !git pull
else:
    !git clone https://github.com/ravidsun/llm-finetune.git
    %cd llm-finetune
print("✅ Repository ready!")

# Step 4: Install Project
print("\n⚙️ Step 4/7: Installing project...")
!pip install -q -e .
print("✅ Project installed!")

# Step 5: Mount Google Drive
print("\n💾 Step 5/7: Mounting Google Drive (for persistence)...")
from google.colab import drive
drive.mount('/content/drive')

# Create directories in Drive
!mkdir -p /content/drive/MyDrive/llm-finetune/data/input
!mkdir -p /content/drive/MyDrive/llm-finetune/output
!mkdir -p /content/drive/MyDrive/llm-finetune/configs
print("✅ Google Drive mounted!")

# Step 6: Set up Hugging Face Token
print("\n🔑 Step 6/7: Setting up Hugging Face authentication...")
from getpass import getpass
hf_token = getpass("Enter your Hugging Face token (get from https://huggingface.co/settings/tokens): ")
os.environ['HF_TOKEN'] = hf_token
!huggingface-cli login --token {hf_token}
print("✅ Authenticated with Hugging Face!")

# Step 7: Verify Installation
print("\n✔️ Step 7/7: Verifying installation...")
!python -m finetune_project --help
print("\n" + "="*60)
print("✅ SETUP COMPLETE! You're ready to fine-tune!")
print("="*60)
print("\n📖 Next steps:")
print("1. Upload your JSONL training file (Cell 2)")
print("2. Create config file (Cell 3)")
print("3. Start training! (Cell 4)")
```

### Cell 2: Upload Your Data

**Option A: Upload Existing JSONL**

```python
# ==========================================
# Upload Your Existing JSONL Training File
# ==========================================

from google.colab import files
import os

print("📤 Upload your JSONL training file...")
print("Expected format: {\"instruction\": \"...\", \"input\": \"...\", \"output\": \"...\"}")

# Create directory
!mkdir -p data/input

# Upload file
uploaded = files.upload()

# Move to correct location
for filename in uploaded.keys():
    !mv "{filename}" data/input/train.jsonl
    print(f"\n✅ Uploaded: {filename}")
    print(f"📍 Saved to: data/input/train.jsonl")

# Verify file
print("\n📊 File info:")
!wc -l data/input/train.jsonl
print("\nFirst 3 lines:")
!head -n 3 data/input/train.jsonl

print("\n✅ Data ready! Continue to Cell 3...")
```

**Option B: Create Sample Data (for testing)**

```python
# ==========================================
# Create Sample Training Data
# ==========================================

import json
import os

# Sample data for testing
sample_data = [
    {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
    {"instruction": "Explain neural networks", "input": "", "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process and transform data."},
    {"instruction": "What is deep learning?", "input": "", "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input."},
    {"instruction": "Define supervised learning", "input": "", "output": "Supervised learning is a machine learning approach where the model is trained on labeled data, learning to map inputs to known outputs."},
    {"instruction": "What is a transformer?", "input": "", "output": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data, forming the basis of modern language models."}
]

# Save to file
os.makedirs('data/input', exist_ok=True)
with open('data/input/train.jsonl', 'w') as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Created sample data: {len(sample_data)} examples")
!head -n 3 data/input/train.jsonl
```

### Cell 3: Create Configuration

```python
# ==========================================
# Create Training Configuration
# ==========================================

config_yaml = """
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  use_qlora: true  # Required for Colab T4

data:
  input_path: "data/input"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false  # Already have JSONL

  augmentation:
    enabled: false  # Set true to augment data

training:
  # Colab T4 optimized settings
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  max_seq_length: 512  # Small for T4
  gradient_checkpointing: true
  fp16: false
  bf16: false  # T4 doesn't support bf16
  output_dir: "output"
  save_strategy: "epoch"
  save_total_limit: 2
  logging_steps: 10
  report_to: []
"""

# Save config
with open('config.yaml', 'w') as f:
    f.write(config_yaml)

print("✅ Config created: config.yaml")
print("\n📄 Configuration:")
!cat config.yaml
```

### Cell 4: Start Training

```python
# ==========================================
# Start Training
# ==========================================

print("🚀 Starting training...")
print("⏱️ This may take 30 minutes - 2 hours depending on data size")
print("📊 Training progress will be shown below\n")

# Train the model
!python -m finetune_project train --config config.yaml

print("\n✅ Training complete!")
print("📁 Model saved to: output/")
```

### Cell 5: Save Your Model

```python
# ==========================================
# Save Model to Google Drive
# ==========================================

import shutil
from google.colab import files

print("💾 Saving model...")

# IMPORTANT: Save to Google Drive (session will disconnect!)
!cp -r output /content/drive/MyDrive/llm-finetune/
print("✅ Model saved to Google Drive!")
print("📂 Location: /content/drive/MyDrive/llm-finetune/output/")

# Optional: Create ZIP for download
print("\n📦 Creating ZIP for download...")
!zip -r my-finetuned-model.zip output/
print("✅ ZIP created: my-finetuned-model.zip")

# Optional: Download now
download = input("\n📥 Download ZIP now? (y/n): ")
if download.lower() == 'y':
    files.download('my-finetuned-model.zip')
    print("✅ Download started!")

print("\n" + "="*60)
print("✅ ALL DONE! Your model is saved and ready!")
print("="*60)
```

---

## Step-by-Step Setup

If you prefer manual control, follow these detailed steps.

### Step 1: Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free) or **A100** (Pro+)
4. Click **Save**

**Verify GPU**:
```python
!nvidia-smi
```

Expected: `Tesla T4` with 15GB VRAM

### Step 2: Install Dependencies

```python
# Core libraries
!pip install -q -U transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.27.0
!pip install -q -U peft>=0.10.0 trl>=0.8.0 bitsandbytes>=0.43.0

# LangChain for document processing
!pip install -q -U langchain>=0.2.0 langchain-community>=0.2.0

# Document loaders
!pip install -q -U pymupdf>=1.24.0 python-docx>=1.1.0

# Verify
import torch, transformers, peft
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

### Step 3: Clone Repository

```python
# Clone
!git clone https://github.com/ravidsun/llm-finetune.git
%cd llm-finetune

# Install project
!pip install -q -e .

# Verify
!python -m finetune_project --help
```

**Troubleshooting**: If you see `[Errno 2] No such file or directory`, the clone failed. Check internet and retry.

### Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Create directories
!mkdir -p /content/drive/MyDrive/llm-finetune/data/input
!mkdir -p /content/drive/MyDrive/llm-finetune/output
```

### Step 5: Authenticate

```python
import os
from getpass import getpass

# Hugging Face token (required)
hf_token = getpass("Enter your HF token: ")
os.environ['HF_TOKEN'] = hf_token
!huggingface-cli login --token {hf_token}
```

Get token at: https://huggingface.co/settings/tokens

---

## Data Preparation

### Option A: Upload Existing JSONL (Recommended)

If you already have JSONL files:

```python
from google.colab import files

# Create directory
!mkdir -p data/input

# Upload
print("Upload your JSONL file...")
uploaded = files.upload()

# Move to standard location
!mv *.jsonl data/input/train.jsonl

# Verify
!head -n 3 data/input/train.jsonl
!wc -l data/input/train.jsonl
```

**Expected format**:
```jsonl
{"instruction": "What is Python?", "input": "", "output": "Python is a programming language..."}
{"instruction": "Explain loops", "input": "in Python", "output": "Loops in Python..."}
```

### Option B: Upload PDFs/DOCX

For documents that need processing:

```python
# Upload files
uploaded = files.upload()

# Move to data directory
!mkdir -p data
!mv *.pdf data/ 2>/dev/null || true
!mv *.docx data/ 2>/dev/null || true

# List uploaded files
!ls -lh data/
```

Then enable LangChain in config:
```yaml
data:
  input_path: "data"
  input_type: "pdf"  # or "docx"
  langchain:
    enabled: true
```

### Option C: Download from URL

```python
# Download JSONL from URL
!wget -O data/input/train.jsonl "https://your-url.com/data.jsonl"

# Or from Hugging Face
from datasets import load_dataset
dataset = load_dataset("your-username/dataset")
dataset['train'].to_json('data/input/train.jsonl')
```

---

## Training

### Create Config File

```python
config_yaml = """
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  input_path: "data/input"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false

  augmentation:
    enabled: false

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  max_seq_length: 512
  gradient_checkpointing: true
  output_dir: "output"
  save_strategy: "epoch"
  save_total_limit: 2
  logging_steps: 10
"""

with open('config.yaml', 'w') as f:
    f.write(config_yaml)
```

### Start Training

```python
# Train
!python -m finetune_project train --config config.yaml
```

**Expected output**:
```
Step 1/150 | Loss: 2.45
Step 10/150 | Loss: 1.82
Step 20/150 | Loss: 1.34
...
Training complete!
```

### Monitor Training

View real-time progress:
```python
# In a separate cell while training runs
!tail -f output/trainer_log.txt
```

Or use TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir output/runs
```

---

## Testing & Evaluation

### Test the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "output")

# Test prompt
prompt = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Compare Base vs Fine-tuned

```python
# Test same prompt on base model
base_outputs = base_model.generate(**inputs, max_new_tokens=256)
base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

print("=" * 60)
print("BASE MODEL:")
print(base_response)
print("\n" + "=" * 60)
print("FINE-TUNED MODEL:")
print(response)
print("=" * 60)
```

---

## Saving Your Model

### Option 1: Save to Google Drive (Recommended)

**CRITICAL**: Colab sessions disconnect! Always save to Drive.

```python
# Copy entire output to Drive
!cp -r output /content/drive/MyDrive/llm-finetune/

print("✅ Model saved to Google Drive!")
print("Location: /content/drive/MyDrive/llm-finetune/output/")
```

### Option 2: Download as ZIP

```python
from google.colab import files

# Create ZIP
!zip -r my-model.zip output/

# Download
files.download('my-model.zip')
```

### Option 3: Upload to Hugging Face

```python
from huggingface_hub import HfApi

# Login (if not already)
!huggingface-cli login

# Upload
api = HfApi()
api.upload_folder(
    folder_path="output",
    repo_id="your-username/your-model-name",
    repo_type="model"
)

print("✅ Model uploaded to Hugging Face!")
```

---

## Troubleshooting

### GPU Not Available

**Error**: `CUDA not available`

**Solution**:
1. Runtime → Change runtime type
2. Hardware accelerator: GPU
3. Save and reconnect

### Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32

# Reduce sequence length
training:
  max_seq_length: 256

# Use smaller model
model:
  model_name: "Qwen/Qwen2.5-3B-Instruct"
```

### Directory Not Found

**Error**: `[Errno 2] No such file or directory: 'llm-finetune'`

**Solution**:
```python
# Clone repository first
!git clone https://github.com/ravidsun/llm-finetune.git
%cd llm-finetune
```

### GitHub Authentication Error

**Error**: `fatal: could not read Username for 'https://github.com'`

**Solution**: Skip GitHub push, save to Drive instead:
```python
# Don't use git push in Colab
# Instead, save to Drive
!cp -r output /content/drive/MyDrive/llm-finetune/
```

### Session Disconnected

**Solution**: Resume from checkpoint
```python
# Find checkpoint
!ls output/

# Resume training
!python -m finetune_project train \
    --config config.yaml \
    --resume-from-checkpoint output/checkpoint-100
```

### Installation Fails

**Solution**: Clear cache and reinstall
```python
!pip cache purge
!pip install --upgrade --force-reinstall transformers peft trl
```

---

## Best Practices

### 1. Always Save to Drive

```python
# At regular intervals during long training
!cp -r output /content/drive/MyDrive/llm-finetune/backup-$(date +%s)/
```

### 2. Use Checkpointing

```yaml
training:
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
```

### 3. Monitor GPU Usage

```python
# Check GPU utilization
!nvidia-smi
```

### 4. Keep Session Alive (Optional)

```python
# Run in background to prevent disconnect
from IPython.display import Javascript
import time

def keep_alive():
    while True:
        display(Javascript('document.title="Active"'))
        time.sleep(60)

# Note: This may not always work
```

---

## Additional Resources

- **Full Documentation**: [GitHub Repository](https://github.com/ravidsun/llm-finetune)
- **Troubleshooting Guide**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Using Existing JSONL**: [USING_EXISTING_JSONL.md](USING_EXISTING_JSONL.md)
- **Config Examples**: [configs/](../configs/)

---

## Summary

### What You Learned

✅ Setting up Google Colab with GPU
✅ Installing dependencies and cloning repository
✅ Uploading and preparing training data
✅ Configuring fine-tuning parameters
✅ Training a 7B model with LoRA/QLoRA
✅ Testing and evaluating your model
✅ Saving models to Drive or downloading
✅ Troubleshooting common issues

### Next Steps

1. **Experiment with different models**: Try Llama, Mistral, or Phi
2. **Adjust hyperparameters**: Learning rate, batch size, epochs
3. **Add more data**: Better data = better model
4. **Deploy your model**: Use vLLM, llama.cpp, or HuggingFace

**Good luck with your fine-tuning!** 🚀
