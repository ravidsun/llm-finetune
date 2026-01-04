# RunPod Fine-Tuning Guide (Automated)

Complete automated setup for fine-tuning LLMs on RunPod. Most steps are automated with copy-paste commands.

⏱️ **Total Time**: 30-45 minutes (mostly automated)
💰 **Cost**: $5-20 for training (pay-per-hour)
🎯 **Difficulty**: Beginner-friendly

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Create RunPod Instance](#step-1-create-runpod-instance)
3. [Step 2: Automated Setup](#step-2-automated-setup)
4. [Step 3: Upload Your Data](#step-3-upload-your-data)
5. [Step 4: Start Training](#step-4-start-training)
6. [Step 5: Monitor Training](#step-5-monitor-training)
7. [Step 6: Save Your Model](#step-6-save-your-model)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### What You Need

✅ **RunPod Account** - [Sign up here](https://runpod.io) (free)
✅ **Credits** - Add $10-15 to your account
✅ **HuggingFace Token** - [Get from here](https://huggingface.co/settings/tokens)
✅ **Training Data** - JSONL files with your data

### Optional
- **Anthropic API Key** - Only if using QA generation
- **W&B API Key** - Only if using Weights & Biases tracking

---

## Step 1: Create RunPod Instance

### 1.1 Choose Your GPU

Log into [RunPod](https://runpod.io/console/pods) and deploy a new pod:

| Model Size | Recommended GPU | VRAM | Cost/Hour |
|------------|----------------|------|-----------|
| **7B with QLoRA** | RTX 4090 | 24GB | ~$0.44 |
| **14B with QLoRA** | A40 | 46GB | ~$0.39 |
| **32B with QLoRA** | A100 40GB | 40GB | ~$1.89 |

### 1.2 Pod Configuration

1. Click **"Deploy"** on your chosen GPU
2. **Template**: Select "RunPod PyTorch 2.1" or "PyTorch"
3. **Container Disk**: 30 GB minimum
4. **Volume Disk**: 50 GB (recommended for model storage)
5. **Expose Ports**: Enable HTTP (for Jupyter if needed)
6. Click **"Deploy On-Demand"**

### 1.3 Connect to Pod

Once deployed:
1. Click **"Connect"** → **"Start Web Terminal"** or **"Connect via SSH"**
2. You'll see a terminal - you're ready to go!

---

## Step 2: Automated Setup

### 2.1 One-Command Setup (Recommended)

Copy and paste this **single command** to set up everything:

```bash
curl -fsSL https://raw.githubusercontent.com/ravidsun/llm-finetune/master/scripts/runpod_auto_setup.sh | bash
```

This script will automatically:
- ✅ Install all dependencies (PyTorch, Transformers, LangChain)
- ✅ Clone the repository
- ✅ Set up directory structure
- ✅ Verify GPU access
- ✅ Install the project

**Time**: ~5-8 minutes

### 2.2 Manual Setup (Alternative)

If you prefer step-by-step:

```bash
# Update system
apt-get update

# Clone repository
cd /workspace
git clone https://github.com/ravidsun/llm-finetune.git
cd llm-finetune

# Install dependencies
pip install -e .

# Verify installation
python -m finetune_project --help
```

### 2.3 Set Environment Variables

Set your HuggingFace token (required):

```bash
# Interactive prompt
export HF_TOKEN=$(python3 -c "from getpass import getpass; print(getpass('Enter your HuggingFace token: '))")

# Or set directly (not recommended - visible in history)
export HF_TOKEN="your_token_here"

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN
```

Optional - set other API keys:

```bash
# For QA generation (optional)
export ANTHROPIC_API_KEY="your_anthropic_key"

# For experiment tracking (optional)
export WANDB_API_KEY="your_wandb_key"
```

---

## Step 3: Upload Your Data

### Option A: Upload JSONL Files (Recommended)

**Method 1: Web Upload via RunPod UI**

1. In RunPod UI, click **"Connect"** → **"Upload Files"**
2. Navigate to `/workspace/llm-finetune/data/input/`
3. Upload your `.jsonl` files

**Method 2: SCP from Local Machine**

```bash
# From your local machine
scp /path/to/your/train.jsonl root@<POD-IP>:/workspace/llm-finetune/data/input/

# Get POD-IP from RunPod UI (Connect → SSH)
```

**Method 3: Download from URL**

```bash
# In RunPod terminal
cd /workspace/llm-finetune/data/input
wget https://your-url.com/train.jsonl
```

### Option B: Use Sample Data (For Testing)

```bash
cd /workspace/llm-finetune

# Run automated data creation
python scripts/create_sample_data.py --output data/input/train.jsonl --samples 100

# Verify
head -n 3 data/input/train.jsonl
```

### Verify Your Data

```bash
# Check file exists and format
ls -lh /workspace/llm-finetune/data/input/*.jsonl
head -n 3 /workspace/llm-finetune/data/input/*.jsonl
wc -l /workspace/llm-finetune/data/input/*.jsonl
```

---

## Step 4: Start Training

### 4.1 Quick Start (Fully Automated)

Use the automated training script:

```bash
cd /workspace/llm-finetune

# One-command training with sensible defaults
bash scripts/runpod_quick_train.sh
```

This will:
1. Auto-detect your GPU
2. Generate optimized config
3. Process your data
4. Start training in tmux
5. Show you how to monitor

**Time**: Training starts in ~2 minutes

### 4.2 Custom Configuration (Advanced)

If you want to customize settings:

```bash
cd /workspace/llm-finetune

# Generate config with your preferences
python -m finetune_project init \
    --output config.yaml \
    --template existing_jsonl

# Edit config (optional)
nano config.yaml

# Process data
python -m finetune_project prepare-data --config config.yaml

# Start training in tmux
tmux new -s training
python -m finetune_project train --config config.yaml

# Detach from tmux: Ctrl+B, then D
```

### 4.3 Training Configuration Examples

**For 7B Model on RTX 4090 (24GB VRAM):**
```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "/workspace/output"
```

**For 14B Model on A40 (46GB VRAM):**
```yaml
model:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  lora_rank: 32
  lora_alpha: 64
  use_qlora: true

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.5e-4
  max_seq_length: 4096
  output_dir: "/workspace/output"
```

---

## Step 5: Monitor Training

### 5.1 Attach to Training Session

```bash
# Attach to tmux session
tmux attach -t training

# Detach again: Ctrl+B, then D
```

### 5.2 Check Training Progress

```bash
# View training logs
tail -f /workspace/llm-finetune/output/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check latest checkpoint
ls -lth /workspace/llm-finetune/output/checkpoint-*/ | head -5
```

### 5.3 Monitor with TensorBoard (Optional)

```bash
# In a new tmux window
tmux new -s tensorboard
tensorboard --logdir /workspace/llm-finetune/output --port 6006

# Access via RunPod's exposed port
```

### Training Progress Indicators

✅ **Good Signs:**
- Loss decreasing consistently
- GPU utilization > 80%
- No OOM errors
- Checkpoints saving regularly

⚠️ **Warning Signs:**
- Loss not decreasing after 100 steps
- GPU utilization < 50%
- Frequent OOM errors → Reduce batch size

---

## Step 6: Save Your Model

### 6.1 Save to RunPod Volume (Persistent)

```bash
# After training completes
cd /workspace/llm-finetune

# Copy to persistent volume
cp -r output /workspace/my-finetuned-model-$(date +%Y%m%d)

# Verify
ls -lh /workspace/my-finetuned-model-*/
```

### 6.2 Download to Local Machine

**Method 1: SCP (Recommended for large files)**

```bash
# From your local machine
scp -r root@<POD-IP>:/workspace/llm-finetune/output ./my-model

# Or create zip first
# On RunPod:
cd /workspace/llm-finetune
tar -czf my-model.tar.gz output/

# Then download:
# scp root@<POD-IP>:/workspace/llm-finetune/my-model.tar.gz .
```

**Method 2: Upload to HuggingFace Hub**

```bash
cd /workspace/llm-finetune

# Upload adapter
python -m finetune_project upload \
    --model-path output \
    --repo-id your-username/your-model-name \
    --token $HF_TOKEN
```

**Method 3: Google Drive (Using rclone)**

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive (follow prompts)
rclone config

# Upload
rclone copy /workspace/llm-finetune/output gdrive:llm-models/my-model
```

### 6.3 Merge Adapter (Optional)

To create a standalone model (not requiring base model):

```bash
cd /workspace/llm-finetune

python -m finetune_project merge-adapter \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter output \
    --output /workspace/merged-model

# Upload merged model
python -m finetune_project upload \
    --model-path /workspace/merged-model \
    --repo-id your-username/your-merged-model \
    --token $HF_TOKEN
```

---

## Troubleshooting

### Out of Memory Errors

```bash
# Edit your config.yaml
nano config.yaml

# Reduce these values:
# per_device_train_batch_size: 1  # Was 2
# gradient_accumulation_steps: 16  # Was 8
# max_seq_length: 1024  # Was 2048
```

### Training Not Starting

```bash
# Check GPU is available
nvidia-smi

# Check data is present
ls -lh /workspace/llm-finetune/data/input/*.jsonl

# Check logs
cat /workspace/llm-finetune/output/training.log
```

### Pod Disconnected

```bash
# Don't worry! Training continues in tmux
# Just reconnect to pod and attach:
tmux attach -t training
```

### Need to Resume Training

```bash
# Find last checkpoint
ls -lth /workspace/llm-finetune/output/checkpoint-*/

# Resume from checkpoint
python -m finetune_project train \
    --config config.yaml \
    --resume-from-checkpoint output/checkpoint-500
```

---

## Cost Management

### Minimize Costs

1. **Use Spot Instances** - 50-70% cheaper (may be interrupted)
2. **Stop Pod When Not Training** - Only pay when running
3. **Use Appropriate GPU** - Don't overpay for A100 if RTX 4090 works
4. **Optimize Batch Size** - Faster training = lower cost
5. **Monitor Training** - Stop if loss plateaus early

### Estimated Costs

| Scenario | GPU | Duration | Cost |
|----------|-----|----------|------|
| 7B, 1K samples | RTX 4090 | 1-2 hours | $0.44-0.88 |
| 7B, 10K samples | RTX 4090 | 5-8 hours | $2.20-3.52 |
| 14B, 10K samples | A40 | 8-12 hours | $3.12-4.68 |

---

## Next Steps

After training completes:

1. **Test Your Model** - See [Phase 4: Evaluation](docs/PHASE4_EVALUATION.md)
2. **Deploy** - See [Phase 5: Deployment](docs/PHASE5_DEPLOYMENT.md)
3. **Share** - Upload to HuggingFace Hub
4. **Iterate** - Adjust hyperparameters and retrain

---

## Quick Reference Commands

```bash
# Connect to training session
tmux attach -t training

# Check GPU usage
nvidia-smi

# View training logs
tail -f /workspace/llm-finetune/output/training.log

# List checkpoints
ls -lth /workspace/llm-finetune/output/checkpoint-*/

# Stop training (if needed)
tmux kill-session -t training

# Download model
scp -r root@<POD-IP>:/workspace/llm-finetune/output ./my-model
```

---

## Support

- **Documentation**: [Full Docs](docs/README.md)
- **Troubleshooting**: [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- **Issues**: [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)

**Good luck with your fine-tuning!** 🚀
