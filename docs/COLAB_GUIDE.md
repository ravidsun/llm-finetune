# Google Colab Fine-Tuning Guide

Complete guide for fine-tuning LLMs using Google Colab's free/paid GPUs.

## Overview

Google Colab provides free GPU access (T4) and paid options (A100, V100) that work well for fine-tuning smaller models (7B) with LoRA/QLoRA.

### Colab GPU Options

| Tier | GPU | VRAM | Cost | Best For |
|------|-----|------|------|----------|
| Free | T4 | 15GB | Free | 7B models with QLoRA |
| Colab Pro | T4/V100 | 15-16GB | $10/month | 7B models, longer sessions |
| Colab Pro+ | V100/A100 | 16-40GB | $50/month | 7B-13B models, priority access |

### Limitations

⚠️ **Important Colab Constraints**:
- **Session timeout**: 12 hours (free), 24 hours (Pro)
- **Idle timeout**: 90 minutes (free), longer (Pro)
- **Disk space**: ~80GB available
- **RAM**: 12-32GB depending on tier
- **No persistent storage** - Download models after training!

## Quick Start

### Option 1: Use Pre-made Notebook (Easiest)

1. Open the notebook: [llm_finetune_colab.ipynb](../notebooks/llm_finetune_colab.ipynb)
2. Click "Open in Colab" badge
3. Runtime → Change runtime type → GPU (T4/A100)
4. Run cells sequentially
5. Download trained model before session ends

### Option 2: Manual Setup (Full Control)

Follow the steps below for complete control over the process.

---

## Step-by-Step Colab Setup

### Step 1: Configure Runtime

```python
# Check GPU availability
!nvidia-smi

# Expected output: Tesla T4, V100, or A100
```

If no GPU:
1. Runtime → Change runtime type
2. Hardware accelerator: GPU
3. GPU type: T4 (free) or A100 (Pro+)
4. Click Save

### Step 2: Install Dependencies

```python
# Install core libraries (takes ~3-5 minutes)
!pip install -q -U \
    torch torchvision torchaudio \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.27.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    bitsandbytes>=0.43.0 \
    safetensors>=0.4.0

# Install LangChain for document processing
!pip install -q -U \
    langchain>=0.2.0 \
    langchain-core>=0.2.0 \
    langchain-community>=0.2.0 \
    langchain-text-splitters>=0.2.0

# Install document loaders
!pip install -q -U \
    pymupdf>=1.24.0 \
    python-docx>=1.1.0

# Install CLI tools
!pip install -q -U \
    typer[all]>=0.9.0 \
    rich>=13.0.0 \
    pyyaml>=6.0

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Step 3: Clone Repository

```python
# Clone the repository
!git clone https://github.com/ravidsun/llm-finetune.git
%cd llm-finetune

# Install the package
!pip install -q -e .

# Verify
!python -m finetune_project --help
```

### Step 4: Mount Google Drive (Optional but Recommended)

Store data and outputs in Google Drive for persistence:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create directories in Drive
!mkdir -p /content/drive/MyDrive/llm-finetune/data
!mkdir -p /content/drive/MyDrive/llm-finetune/output
!mkdir -p /content/drive/MyDrive/llm-finetune/configs
```

### Step 5: Set Up Authentication

```python
import os
from getpass import getpass

# Hugging Face token (required for gated models)
hf_token = getpass("Enter your Hugging Face token: ")
os.environ['HF_TOKEN'] = hf_token

# Login to Hugging Face
!huggingface-cli login --token {hf_token}

# Optional: Anthropic API key (for QA generation)
# anthropic_key = getpass("Enter your Anthropic API key (or press Enter to skip): ")
# if anthropic_key:
#     os.environ['ANTHROPIC_API_KEY'] = anthropic_key

# Optional: WandB for experiment tracking
# wandb_key = getpass("Enter your WandB API key (or press Enter to skip): ")
# if wandb_key:
#     os.environ['WANDB_API_KEY'] = wandb_key
#     !wandb login {wandb_key}
```

---

## Data Preparation in Colab

### Option A: Upload Local Files

```python
from google.colab import files

# Upload JSON/JSONL file
uploaded = files.upload()

# Move to data directory
!mkdir -p data
!mv *.jsonl data/ 2>/dev/null || true
!mv *.json data/ 2>/dev/null || true
!mv *.pdf data/ 2>/dev/null || true
```

### Option B: Use Sample Dataset

```python
# Create sample data
sample_data = [
    {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
    {"instruction": "Explain neural networks", "input": "", "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process and transform data."},
    {"instruction": "What is deep learning?", "input": "", "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input."}
]

import json
with open('data/sample.jsonl', 'w') as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')

print("Sample data created!")
```

### Option C: Download from URL

```python
# Download dataset from URL
!wget -O data/dataset.jsonl "YOUR_DATASET_URL"

# Or from Hugging Face
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset")
dataset['train'].to_json('data/train.jsonl')
```

---

## Configuration for Colab

### Create Colab-Optimized Config

```python
# Create config file optimized for Colab T4 (15GB VRAM)
config_yaml = """
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # Or unsloth/Llama-3.2-3B-Instruct
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  use_qlora: true  # IMPORTANT: Use 4-bit for Colab

data:
  input_path: "data"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false  # Set true if using PDFs
    qa_generation_enabled: false

  augmentation:
    enabled: false

training:
  # Colab-optimized settings for T4 (15GB)
  num_epochs: 3
  per_device_train_batch_size: 1  # Small batch for T4
  gradient_accumulation_steps: 16  # Effective batch = 16

  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"

  max_seq_length: 512  # Reduce for Colab

  # Memory optimizations
  gradient_checkpointing: true
  fp16: false
  bf16: false  # T4 doesn't support bf16

  # Output
  output_dir: "output"
  save_strategy: "epoch"  # Save less frequently
  save_total_limit: 2  # Keep only 2 checkpoints

  logging_steps: 10
  report_to: []  # Or ["tensorboard"] or ["wandb"]

  evaluation_strategy: "no"
"""

# Save config
with open('config.yaml', 'w') as f:
    f.write(config_yaml)

print("Config created: config.yaml")

# Or save to Google Drive
# with open('/content/drive/MyDrive/llm-finetune/configs/config.yaml', 'w') as f:
#     f.write(config_yaml)
```

### For Colab Pro+ with A100 (40GB)

```python
config_yaml_a100 = """
model:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  lora_rank: 32
  lora_alpha: 64
  use_qlora: false  # Can use full precision on A100

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  bf16: true  # A100 supports bf16
  gradient_checkpointing: false
"""
```

---

## Training in Colab

### Method 1: Using CLI

```python
# Prepare data
!python -m finetune_project prepare-data --config config.yaml

# Start training
!python -m finetune_project train --config config.yaml
```

### Method 2: Python API (More Control)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Load model with 4-bit quantization
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset('json', data_files='data/sample.jsonl', split='train')

# Format function
def format_instruction(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example['input']:
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_instruction)

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_steps=-1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to=[],
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

# Train
print("Starting training...")
trainer.train()

print("Training complete! Saving model...")
trainer.save_model("output")
tokenizer.save_pretrained("output")
```

---

## Monitoring Training

### Option 1: Watch Logs

Training progress will show in cell output:
- Current step / total steps
- Training loss
- GPU memory usage

### Option 2: TensorBoard

```python
# Load TensorBoard
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir output/runs

# Training will log automatically if report_to=["tensorboard"]
```

### Option 3: W&B (Recommended)

```python
# Already installed, just enable in config
# report_to: ["wandb"]

# Or start manually
import wandb
wandb.init(project="llm-finetune", name="colab-run-1")
```

---

## Save Model to Google Drive

**CRITICAL**: Colab sessions disconnect. Save to Drive!

```python
# After training completes
import shutil

# Copy output to Google Drive
!cp -r output /content/drive/MyDrive/llm-finetune/

print("Model saved to Google Drive!")

# Or create a zip for download
!zip -r model.zip output/
files.download('model.zip')
```

---

## Testing the Model

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "output")

# Test
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

---

## Download Model

### Method 1: Direct Download

```python
from google.colab import files

# Zip and download
!zip -r my-lora-adapter.zip output/
files.download('my-lora-adapter.zip')
```

### Method 2: Upload to Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="output",
    repo_id="your-username/your-model-name",
    repo_type="model",
    token=hf_token
)

print("Model uploaded to Hugging Face!")
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 32

# Reduce sequence length
max_seq_length: 256

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller model
model_name: "unsloth/Llama-3.2-3B-Instruct"
```

### Session Disconnected

```python
# Reconnect and resume from checkpoint
!python -m finetune_project train \
    --config config.yaml \
    --resume-from-checkpoint output/checkpoint-XXX
```

### Slow Training

```python
# Check GPU utilization
!nvidia-smi

# Increase batch size if VRAM allows
per_device_train_batch_size: 2

# Use packing
packing: true
```

### Installation Fails

```python
# Clear and reinstall
!pip cache purge
!pip install --upgrade --force-reinstall transformers peft trl
```

---

## Best Practices for Colab

### 1. Keep Session Alive

```python
# Run this in a separate cell
from IPython.display import Javascript
import time

def keep_alive():
    while True:
        display(Javascript('if (document.hidden) { document.title = "Active"; }'))
        time.sleep(60)

# Run in background (optional)
# import threading
# thread = threading.Thread(target=keep_alive)
# thread.start()
```

### 2. Checkpoint Frequently

```python
training:
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 2
```

### 3. Use Smaller Models

Best models for Colab T4:
- `unsloth/Llama-3.2-3B-Instruct` (3B - fastest)
- `Qwen/Qwen2.5-7B-Instruct` (7B - good quality)
- `meta-llama/Llama-3.2-8B-Instruct` (8B - max for T4)

### 4. Monitor VRAM

```python
# Check VRAM usage
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## Complete Workflow Example

```python
# 1. Setup
!pip install -q transformers peft trl datasets bitsandbytes accelerate
!git clone https://github.com/ravidsun/llm-finetune.git
%cd llm-finetune
!pip install -q -e .

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Prepare data (upload or create)
!mkdir -p data
# ... upload your data ...

# 4. Create config
# ... create config.yaml ...

# 5. Authenticate
import os
from getpass import getpass
os.environ['HF_TOKEN'] = getpass("HF Token: ")

# 6. Prepare data
!python -m finetune_project prepare-data --config config.yaml

# 7. Train
!python -m finetune_project train --config config.yaml

# 8. Save to Drive
!cp -r output /content/drive/MyDrive/llm-finetune/

# 9. Test
# ... test your model ...

# 10. Download or upload to HF
!zip -r model.zip output/
from google.colab import files
files.download('model.zip')
```

---

## Time & Cost Estimates

### Colab Free (T4, 15GB)

| Dataset Size | Model | Training Time | Cost |
|--------------|-------|---------------|------|
| 1K samples | 7B | 30-60 min | Free |
| 10K samples | 7B | 3-6 hours | Free |
| 100K samples | 7B | 24+ hours | Not feasible* |

*Free tier disconnects after 12 hours

### Colab Pro+ (A100, 40GB)

| Dataset Size | Model | Training Time | Cost |
|--------------|-------|---------------|------|
| 1K samples | 7B | 15-30 min | ~$2 |
| 10K samples | 7B | 2-4 hours | ~$8 |
| 10K samples | 14B | 4-8 hours | ~$16 |

---

## Next Steps

After training in Colab:
- Download model to local machine
- Deploy with [PHASE5_DEPLOYMENT.md](PHASE5_DEPLOYMENT.md)
- Or upload to Hugging Face for easy access

## Resources

- [Official Colab Notebook](../notebooks/llm_finetune_colab.ipynb)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Hugging Face Docs](https://huggingface.co/docs)
