# Google Colab Quick Start Cell

Copy and paste this single cell into a new Colab notebook to get started immediately.

---

## Complete Setup Cell (Copy This!)

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
    print("⚠️ Repository already exists, using existing copy")
    %cd /content/llm-finetune
    !git pull  # Update if already cloned
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
print("1. Upload your JSONL training file (see cell below)")
print("2. Create config file")
print("3. Start training!")
print("\nDocumentation: https://github.com/ravidsun/llm-finetune/tree/master/docs")
```

---

## Upload Data Cell (Copy This Next!)

### Option A: Upload Existing JSONL File

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
!echo "\n📊 File info:"
!wc -l data/input/train.jsonl
!head -n 3 data/input/train.jsonl

print("\n✅ Data ready! Continue to config creation...")
```

### Option B: Create Sample Data

```python
# ==========================================
# Create Sample Training Data (for testing)
# ==========================================

import json
import os

# Create sample data
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

---

## Create Config Cell (Copy This Third!)

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
  # Use existing JSONL files
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
!cat config.yaml
```

---

## Training Cell (Copy This Fourth!)

```python
# ==========================================
# Start Training
# ==========================================

print("🚀 Starting training...")
print("This may take 30 minutes - 2 hours depending on data size\n")

# Train the model
!python -m finetune_project train --config config.yaml

print("\n✅ Training complete!")
```

---

## Save Model Cell (Copy This Last!)

```python
# ==========================================
# Save Model to Google Drive
# ==========================================

import shutil
from google.colab import files

print("💾 Saving model...")

# Save to Google Drive (IMPORTANT!)
!cp -r output /content/drive/MyDrive/llm-finetune/
print("✅ Model saved to Google Drive: /content/drive/MyDrive/llm-finetune/output/")

# Optional: Download as ZIP
print("\n📥 Creating ZIP for download...")
!zip -r my-finetuned-model.zip output/
print("✅ ZIP created!")

# Download (optional)
download = input("Download ZIP now? (y/n): ")
if download.lower() == 'y':
    files.download('my-finetuned-model.zip')

print("\n✅ DONE! Your model is saved and ready to use!")
```

---

## Troubleshooting

### If you see: `fatal: could not read Username for 'https://github.com'`

This happens when trying to push to GitHub. **Solution**: Skip GitHub push and save to Drive instead (see "Save Model Cell" above).

### If you see: `[Errno 2] No such file or directory: 'llm-finetune'`

**Solution**: Run the "Complete Setup Cell" again. The repository wasn't cloned.

### If you see: `CUDA out of memory`

**Solution**: Reduce batch size and sequence length in config:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  max_seq_length: 256  # Reduce from 512
```

---

## Complete Notebook Template

Want all cells in one notebook? Check out:
- [llm_finetune_colab.ipynb](../notebooks/llm_finetune_colab.ipynb) - Ready-to-run notebook

---

## More Help

- **Full Guide**: [COLAB_GUIDE.md](COLAB_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **GitHub**: https://github.com/ravidsun/llm-finetune
