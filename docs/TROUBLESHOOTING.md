# Troubleshooting Guide

Common issues and solutions for LLM fine-tuning on Colab, RunPod, and local machines.

---

## Table of Contents

- [GitHub & Git Issues](#github--git-issues)
- [Memory Issues](#memory-issues)
- [Training Issues](#training-issues)
- [Data Issues](#data-issues)
- [Installation Issues](#installation-issues)
- [Platform-Specific Issues](#platform-specific-issues)

---

## GitHub & Git Issues

### Error: `fatal: could not read Username for 'https://github.com'`

**Platform**: Google Colab

**Cause**: Colab cannot authenticate with GitHub for pushing to repositories.

**Solution 1: Use Personal Access Token (Recommended)**

```python
# 1. Create Personal Access Token (PAT):
#    - Visit: https://github.com/settings/tokens
#    - Click "Generate new token (classic)"
#    - Name: "Colab Access"
#    - Select scope: ☑ repo (full control of private repositories)
#    - Click "Generate token"
#    - COPY THE TOKEN (you won't see it again!)

# 2. Configure authentication in Colab
import os
from getpass import getpass

# Enter credentials
github_username = input("GitHub username: ")
print("\nEnter Personal Access Token (PAT):")
print("Create at: https://github.com/settings/tokens")
github_token = getpass("PAT: ")

# Update email with your actual email
your_email = input("Your email: ")

# Configure git
!git config --global user.name "{github_username}"
!git config --global user.email "{your_email}"

# Set remote URL with token
!git remote set-url origin https://{github_username}:{github_token}@github.com/ravidsun/llm-finetune.git

# Now push works
!git push

print("✅ Successfully configured GitHub authentication!")
```

**Solution 2: Skip GitHub and Save to Drive**

```python
# Save model to Google Drive instead
!cp -r output /content/drive/MyDrive/llm-finetune/output/
print("✅ Model saved to Google Drive!")

# Or download directly
from google.colab import files
!zip -r model.zip output/
files.download('model.zip')
```

**Solution 3: Use GitHub CLI**

```python
# Install GitHub CLI
!type -p curl >/dev/null || (apt update && apt install curl -y)
!curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
!chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
!echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
!apt update && apt install gh -y

# Authenticate (follow prompts)
!gh auth login

# Push
!git push
```

### Error: `Permission denied (publickey)`

**Solution**: Use HTTPS instead of SSH

```bash
# Check current remote
git remote -v

# If using SSH (git@github.com), change to HTTPS
git remote set-url origin https://github.com/ravidsun/llm-finetune.git
```

---

## Memory Issues

### Error: `CUDA out of memory`

**Cause**: Model or batch size too large for GPU VRAM.

**Solution 1: Reduce Batch Size**

```yaml
training:
  per_device_train_batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 32  # Increase to maintain effective batch size
```

**Solution 2: Reduce Sequence Length**

```yaml
training:
  max_seq_length: 512  # Reduce from 2048
```

**Solution 3: Enable Gradient Checkpointing**

```yaml
training:
  gradient_checkpointing: true
```

**Solution 4: Use Smaller Model**

```yaml
model:
  model_name: "Qwen/Qwen2.5-3B-Instruct"  # Instead of 7B
  # or
  model_name: "unsloth/Llama-3.2-3B-Instruct"
```

**Solution 5: Enable QLoRA (if not already)**

```yaml
model:
  use_qlora: true  # 4-bit quantization
  load_in_4bit: true
```

**Full Colab-Optimized Config for T4 (15GB)**

```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  use_qlora: true
  lora_rank: 8  # Reduce from 16
  lora_alpha: 16  # Reduce from 32

data:
  input_path: "data/input"
  output_path: "processed_data"
  input_type: "json"

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  learning_rate: 2.0e-4
  max_seq_length: 256  # Very small for T4
  gradient_checkpointing: true
  fp16: false
  bf16: false  # T4 doesn't support bf16
  output_dir: "output"
```

### Error: `RuntimeError: CUDA error: device-side assert triggered`

**Cause**: Usually padding or token ID issues.

**Solution**:

```python
# Add padding token if missing
tokenizer.pad_token = tokenizer.eos_token

# Or set explicitly
tokenizer.pad_token_id = tokenizer.eos_token_id
```

---

## Training Issues

### Training is Very Slow

**Check GPU Utilization**:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or in Python
!nvidia-smi
```

**Solutions**:

1. **Increase Batch Size** (if memory allows):
   ```yaml
   per_device_train_batch_size: 2  # Increase if VRAM available
   ```

2. **Enable Data Packing**:
   ```python
   trainer = SFTTrainer(
       packing=True,  # Pack multiple samples per sequence
       ...
   )
   ```

3. **Check Data Loading**:
   ```yaml
   training:
     dataloader_num_workers: 4  # Parallelize data loading
   ```

### Training Loss Not Decreasing

**Solutions**:

1. **Check Learning Rate**:
   ```yaml
   training:
     learning_rate: 2.0e-4  # Try 1e-4 or 5e-5
   ```

2. **Increase Training Steps**:
   ```yaml
   training:
     num_epochs: 5  # More epochs
   ```

3. **Verify Data Quality**:
   ```python
   # Check your JSONL file
   import json
   with open('data/input/train.jsonl') as f:
       for i, line in enumerate(f):
           if i < 5:  # Print first 5
               print(json.loads(line))
   ```

### Session Disconnected (Colab)

**Solution**: Resume from checkpoint

```python
# List available checkpoints
!ls output/

# Resume training
!python -m finetune_project train \
    --config config.yaml \
    --resume-from-checkpoint output/checkpoint-500
```

**Prevent Disconnection**:

```python
# Keep session alive (run in separate cell)
from IPython.display import Javascript

def keep_alive():
    while True:
        display(Javascript('document.title="Active"'))
        time.sleep(60)

# Or use Colab Pro for longer sessions
```

---

## Data Issues

### Error: `JSONDecodeError: Expecting value`

**Cause**: Malformed JSON in JSONL file.

**Solution**: Validate JSONL format

```python
import json

# Find the problematic line
with open('data/input/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"❌ Error on line {i}: {e}")
            print(f"Line content: {line[:100]}...")
```

**Fix**: Ensure each line is valid JSON:

```jsonl
{"instruction": "Question", "input": "", "output": "Answer"}
```

NOT:

```jsonl
{instruction: "Question", input: "", output: "Answer"}  ❌ Missing quotes
{"instruction": "Question" "input": "", "output": "Answer"}  ❌ Missing comma
```

### Error: `KeyError: 'instruction'`

**Cause**: JSONL missing required fields.

**Solution**: Verify all entries have required fields

```python
import json

required_fields = ['instruction', 'input', 'output']

with open('data/input/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        missing = [field for field in required_fields if field not in data]
        if missing:
            print(f"Line {i} missing fields: {missing}")
```

### Empty Dataset After Processing

**Check**:

1. **File paths are correct**:
   ```python
   !ls -lh data/input/
   !wc -l data/input/train.jsonl
   ```

2. **Config points to right directory**:
   ```yaml
   data:
     input_path: "data/input"  # Not "data" or "processed_data"
     input_type: "json"  # For JSONL files
   ```

---

## Installation Issues

### Error: `No module named 'transformers'`

**Solution**: Reinstall dependencies

```bash
# Clear cache
pip cache purge

# Reinstall
pip install --upgrade transformers peft trl accelerate bitsandbytes
```

### Error: `ImportError: cannot import name 'AutoGPTQForCausalLM'`

**Solution**: Update to latest versions

```bash
pip install --upgrade auto-gptq optimum
```

### Error: `OSError: We couldn't connect to 'https://huggingface.co'`

**Solution**: Check internet connection or use mirror

```python
# Set HF mirror (China)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Or download model first
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="./models/qwen")
```

---

## Platform-Specific Issues

### Google Colab

**Issue**: Colab disconnects frequently

**Solutions**:
- Use Colab Pro for longer sessions (up to 24 hours)
- Save checkpoints frequently:
  ```yaml
  training:
    save_strategy: "steps"
    save_steps: 100
  ```
- Save to Google Drive regularly:
  ```python
  !cp -r output /content/drive/MyDrive/backups/
  ```

**Issue**: Can't upload large files

**Solutions**:
- Use Google Drive:
  ```python
  !cp /content/drive/MyDrive/data/large.jsonl data/input/
  ```
- Download from URL:
  ```python
  !wget -O data/input/train.jsonl "https://your-url.com/data.jsonl"
  ```

### RunPod

**Issue**: Pod keeps stopping

**Solution**: Check billing and set auto-shutdown

```bash
# Set auto-shutdown when idle (optional)
# In RunPod dashboard: Set idle timeout
```

**Issue**: Can't connect via SSH

**Solution**: Use HTTP/HTTPS ports instead

```bash
# Use RunPod's web terminal or Jupyter
# Or expose ports: 8888 (Jupyter), 6006 (TensorBoard)
```

### Local Machine

**Issue**: `CUDA not available`

**Solution**: Install CUDA toolkit

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with matching CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: Windows path errors

**Solution**: Use forward slashes or raw strings

```yaml
# Windows paths
data:
  input_path: "C:/LLM/data/input"  # Forward slashes
  # or
  input_path: "C:\\LLM\\data\\input"  # Escaped backslashes
```

---

## Getting Help

If you're still stuck:

1. **Check the logs**:
   ```bash
   # Training logs
   cat output/trainer_log.txt

   # System logs
   dmesg | tail -50
   ```

2. **Enable debug mode**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Open an issue**: https://github.com/ravidsun/llm-finetune/issues

Include:
- Error message (full traceback)
- Platform (Colab/RunPod/Local)
- GPU type and VRAM
- Config file
- Python/CUDA versions

---

## Quick Diagnostic Checklist

Run this to gather system info:

```python
import torch
import transformers
import peft
import sys

print("=== System Info ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"\n=== GPU Info ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("\n=== Files ===")
!ls -lh data/input/ 2>/dev/null || echo "No data/input/ directory"
!ls -lh processed_data/ 2>/dev/null || echo "No processed_data/ directory"
!ls -lh output/ 2>/dev/null || echo "No output/ directory"
```

Save this output when reporting issues!
