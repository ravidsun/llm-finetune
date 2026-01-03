# Google Colab: Complete Fine-Tuning Guide

Step-by-step guide for fine-tuning LLMs using Google Colab's free GPU.

## Overview

⏱️ **Total Time**: 45-90 minutes (including training)
💰 **Cost**: FREE (or $10-50/month for Pro/Pro+)
🎯 **Difficulty**: Beginner-friendly
📊 **Best For**: 7B models, small-medium datasets (1K-10K samples)

## Prerequisites

- Google account
- Web browser
- Training data (JSON/JSONL format recommended)
- Hugging Face account and token

## What You'll Accomplish

By the end of this guide, you'll have:
- ✅ Fine-tuned a 7B LLM on your custom data
- ✅ Tested your model with sample prompts
- ✅ Downloaded or uploaded your model
- ✅ Understanding of the full fine-tuning pipeline

---

## Step 1: Open Colab Notebook (1 minute)

### 1.1 Click the Badge

Click this badge to open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ravidsun/llm-finetune/blob/master/notebooks/llm_finetune_colab.ipynb)

### 1.2 Save a Copy (Optional)

To save your progress:
1. Click **File** → **Save a copy in Drive**
2. This creates your own editable copy

---

## Step 2: Enable GPU Runtime (2 minutes)

### 2.1 Change Runtime Type

1. Click **Runtime** in the menu
2. Select **Change runtime type**
3. Under **Hardware accelerator**, select **GPU**
4. Choose GPU type:
   - **T4** - Free tier (15GB VRAM)
   - **A100** - Pro+ only (40GB VRAM)
5. Click **Save**

### 2.2 Verify GPU

Run the first cell to check GPU:

```python
!nvidia-smi
```

**Expected Output**:
```
Tesla T4
Memory: 15360MiB
```

✅ **Success**: You see GPU name and memory
❌ **Failed**: "CUDA driver version is insufficient" → Repeat Step 2.1

---

## Step 3: Install Dependencies (3-5 minutes)

### 3.1 Run Installation Cell

The notebook will install:
- PyTorch with CUDA support
- Transformers, PEFT, TRL
- LangChain for document processing
- BitsAndBytes for 4-bit quantization

**Installation time**: ~3-5 minutes

### 3.2 Verify Installation

Check that libraries are installed:

```python
import transformers, peft, trl, langchain
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
```

✅ **Success**: Version numbers printed
❌ **Failed**: Rerun installation cell

---

## Step 4: Clone Repository (1 minute)

### 4.1 Clone and Install

Run the cell to:
1. Clone the GitHub repository
2. Install the package
3. Verify CLI is working

```python
!git clone https://github.com/ravidsun/llm-finetune.git
%cd llm-finetune
!pip install -q -e .
```

### 4.2 Verify

```python
!python -m finetune_project --help
```

✅ **Success**: Help message displayed
❌ **Failed**: Check for error messages in output

---

## Step 5: Mount Google Drive (2 minutes) - IMPORTANT!

### 5.1 Why Mount Drive?

⚠️ **CRITICAL**: Colab sessions disconnect after 12 hours (free) or 24 hours (Pro). Mounting Drive saves your model permanently!

### 5.2 Mount Drive

Run the cell and follow prompts:

1. Click the link to authorize
2. Select your Google account
3. Copy the authorization code
4. Paste into the input box

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5.3 Create Directories

```python
!mkdir -p /content/drive/MyDrive/llm-finetune/data
!mkdir -p /content/drive/MyDrive/llm-finetune/output
!mkdir -p /content/drive/MyDrive/llm-finetune/configs
```

✅ **Success**: "Mounted at /content/drive" message
❌ **Failed**: Check Google account permissions

---

## Step 6: Authentication (2 minutes)

### 6.1 Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: "colab-finetune"
4. Type: **Read**
5. Click **Generate**
6. Copy the token (starts with `hf_`)

### 6.2 Enter Token

Run the cell and paste your token when prompted:

```python
from getpass import getpass
hf_token = getpass("Enter your Hugging Face token: ")
```

**Security**: Token is hidden as you type

✅ **Success**: "Login successful" message
❌ **Failed**: Check token is correct (starts with `hf_`)

---

## Step 7: Prepare Training Data (5-10 minutes)

### Option A: Upload Your Own Data (Recommended)

#### 7.1 Prepare Data File

Your data should be in JSONL format:

```jsonl
{"instruction": "What is machine learning?", "input": "", "output": "Machine learning is..."}
{"instruction": "Explain neural networks", "input": "", "output": "Neural networks are..."}
```

**Format Requirements**:
- One JSON object per line
- Required fields: `instruction`, `input`, `output`
- `input` can be empty string

#### 7.2 Upload File

Run the upload cell:

```python
from google.colab import files
uploaded = files.upload()
```

1. Click **Choose Files**
2. Select your `.jsonl` file
3. Wait for upload to complete

**Upload time**: ~10 seconds for 1MB

✅ **Success**: File name displayed with checkmark
❌ **Failed**: Check file format is .jsonl

### Option B: Use Sample Data (For Testing)

Skip upload and run the sample data cell:

```python
# Creates sample.jsonl with 5 examples
```

This creates a small test dataset to verify the pipeline works.

### Option C: Download from URL

If your data is hosted online:

```python
!wget -O data/dataset.jsonl "YOUR_URL_HERE"
```

---

## Step 8: Create Configuration (2 minutes)

### 8.1 Understanding the Config

The config controls:
- **Model**: Which base model to use
- **LoRA**: How many parameters to train
- **Training**: Batch size, learning rate, epochs
- **Memory**: Quantization and optimization

### 8.2 Default Config (T4 - 15GB)

Run the config cell to create `config.yaml`:

```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true  # 4-bit quantization

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  max_seq_length: 512  # Reduced for Colab
  gradient_checkpointing: true
```

### 8.3 Customize (Optional)

**Change model** (must fit in 15GB VRAM):
```yaml
model_name: "unsloth/Llama-3.2-3B-Instruct"  # Faster, less VRAM
```

**More epochs** (better quality):
```yaml
num_epochs: 5  # Default is 3
```

**Longer sequences** (if VRAM allows):
```yaml
max_seq_length: 1024  # Default is 512
```

✅ **Success**: `config.yaml` file created
❌ **Failed**: Check YAML syntax

---

## Step 9: Prepare Data (1-2 minutes)

### 9.1 Run Data Preparation

This processes your raw data into training format:

```python
!python -m finetune_project prepare-data --config config.yaml
```

**What happens**:
1. Loads your JSONL file
2. Applies prompt template
3. Tokenizes text
4. Saves to `processed_data/train.jsonl`

### 9.2 Verify Output

Check processed data:

```python
!head -n 2 processed_data/train.jsonl
```

**Expected**: Formatted training examples

✅ **Success**: Processed data file created
❌ **Failed**: Check data file format

---

## Step 10: Train Model (30-60 minutes)

### 10.1 Start Training

⏱️ **Estimated time**:
- 1K samples: ~30-45 minutes
- 5K samples: ~2-3 hours
- 10K samples: ~4-6 hours

Run training cell:

```python
!python -m finetune_project train --config config.yaml
```

### 10.2 Monitor Progress

Watch for these indicators:

**Training Log Output**:
```
[100/300] Loss: 2.345
[200/300] Loss: 1.234
[300/300] Loss: 0.876
```

**Good signs** ✅:
- Loss decreasing steadily
- No "Out of Memory" errors
- GPU utilization >90% (check with `!nvidia-smi`)

**Warning signs** ⚠️:
- Loss not decreasing → Check data quality
- OOM errors → Reduce batch_size to 1
- Loss = NaN → Reduce learning_rate

### 10.3 Keep Session Alive

**IMPORTANT**: Colab disconnects after ~90 minutes of inactivity

To prevent disconnection:
1. **Stay active**: Click in browser occasionally
2. **Use Colab Pro**: Longer idle timeout
3. **Monitor regularly**: Check progress every 30 minutes

If disconnected:
1. Reconnect to runtime
2. Remount Google Drive
3. Resume from checkpoint:
   ```python
   !python -m finetune_project train \
       --config config.yaml \
       --resume-from-checkpoint output/checkpoint-XXX
   ```

---

## Step 11: Save to Google Drive (1 minute)

### 11.1 Copy Output

**CRITICAL**: Do this IMMEDIATELY after training completes!

```python
!cp -r output /content/drive/MyDrive/llm-finetune/
```

### 11.2 Verify Save

```python
!ls -lh /content/drive/MyDrive/llm-finetune/output/
```

**Expected files**:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - Trained weights
- `tokenizer_config.json` - Tokenizer settings

✅ **Success**: Files visible in Drive
❌ **Failed**: Check Drive is mounted, retry copy

---

## Step 12: Test Your Model (5 minutes)

### 12.1 Load Model

Run the test cell to load your trained model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "output")
```

**Load time**: ~1-2 minutes

### 12.2 Test Prompts

Generate responses:

```python
def generate_response(prompt):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate_response("What is machine learning?"))
```

### 12.3 Evaluate Quality

**Check for**:
- ✅ Coherent, relevant responses
- ✅ Follows instruction format
- ✅ Matches your training domain
- ❌ Gibberish or repetition
- ❌ Off-topic responses

**If quality is poor**:
- Train for more epochs
- Use more/better training data
- Increase LoRA rank

---

## Step 13: Download Model (3 minutes)

### Option A: Download as ZIP

```python
from google.colab import files

!zip -r my-model.zip output/
files.download('my-model.zip')
```

**Download size**: ~100-500MB for LoRA adapter

### Option B: Upload to Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="output",
    repo_id="your-username/your-model-name",
    repo_type="model",
    token=hf_token
)
```

**Benefits**:
- Easy sharing
- Version control
- Free hosting
- Easy loading: `PeftModel.from_pretrained("your-username/your-model-name")`

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions** (try in order):

1. **Reduce batch size**:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 32
   ```

2. **Reduce sequence length**:
   ```yaml
   max_seq_length: 256
   ```

3. **Use smaller model**:
   ```yaml
   model_name: "unsloth/Llama-3.2-3B-Instruct"
   ```

4. **Reduce LoRA rank**:
   ```yaml
   lora_rank: 8
   ```

### Issue: Session Disconnected

**Symptoms**: Runtime disconnected, notebook stopped

**Solutions**:

1. **Reconnect**:
   - Runtime → Connect to hosted runtime
   - Rerun all cells EXCEPT training
   - Resume training from checkpoint

2. **Resume from checkpoint**:
   ```python
   !python -m finetune_project train \
       --config config.yaml \
       --resume-from-checkpoint output/checkpoint-100
   ```

3. **Prevent future disconnects**:
   - Use Colab Pro (longer sessions)
   - Keep browser tab active
   - Monitor every 30 minutes

### Issue: Training Too Slow

**Symptoms**: <10 iterations per minute

**Solutions**:

1. **Check GPU usage**:
   ```python
   !nvidia-smi
   ```
   Should show >90% utilization

2. **Reduce data processing**:
   ```yaml
   max_seq_length: 256
   ```

3. **Increase batch size** (if VRAM available):
   ```yaml
   per_device_train_batch_size: 2
   ```

### Issue: Poor Model Quality

**Symptoms**: Incoherent or off-topic responses

**Solutions**:

1. **Train longer**:
   ```yaml
   num_epochs: 5
   ```

2. **Check training data**:
   - Verify format is correct
   - Check for duplicates or errors
   - Ensure sufficient diversity

3. **Increase LoRA rank**:
   ```yaml
   lora_rank: 32
   ```

4. **Adjust learning rate**:
   ```yaml
   learning_rate: 3.0e-4  # Increase if loss not decreasing
   learning_rate: 1.0e-4  # Decrease if loss erratic
   ```

### Issue: Import Errors

**Symptoms**: ModuleNotFoundError

**Solutions**:

1. **Rerun installation cell**
2. **Check Python version**:
   ```python
   !python --version  # Should be 3.10+
   ```
3. **Clear and reinstall**:
   ```python
   !pip cache purge
   !pip install --upgrade transformers peft trl
   ```

### Issue: Data Upload Failed

**Symptoms**: Upload button doesn't work

**Solutions**:

1. **Check file size**: <100MB recommended
2. **Try different browser**: Chrome works best
3. **Use Google Drive instead**:
   - Upload to Drive manually
   - Copy from Drive to Colab:
   ```python
   !cp /content/drive/MyDrive/data.jsonl data/
   ```

---

## Best Practices

### 1. Data Preparation

✅ **Do**:
- Use high-quality, diverse examples
- Keep instructions clear and specific
- Balance dataset across topics
- Remove duplicates and errors

❌ **Don't**:
- Mix different prompt formats
- Include personal/sensitive information
- Use extremely long examples (>2048 tokens)

### 2. Training

✅ **Do**:
- Start with small test run (100 samples)
- Monitor loss during training
- Save checkpoints frequently
- Test early and often

❌ **Don't**:
- Train on unverified data
- Ignore OOM warnings
- Train for too many epochs (overfitting)
- Leave session unmonitored

### 3. Resource Management

✅ **Do**:
- Mount Google Drive for persistence
- Download model immediately after training
- Clear outputs to save memory
- Use appropriate model size for GPU

❌ **Don't**:
- Rely only on Colab storage
- Keep multiple large models loaded
- Train models too large for T4

---

## Next Steps

### After Fine-Tuning

1. **Test thoroughly** with diverse prompts
2. **Compare** with base model performance
3. **Iterate** if quality isn't sufficient
4. **Deploy** using [PHASE5_DEPLOYMENT.md](PHASE5_DEPLOYMENT.md)

### Deployment Options

From Colab, you can deploy to:

- **Local (llama.cpp)**: Download ZIP and run locally
- **Hugging Face Inference**: Upload and use HF endpoints
- **RunPod Serverless**: Upload to HF, deploy on RunPod
- **Custom API**: Download and deploy on your server

See [PHASE5_DEPLOYMENT.md](PHASE5_DEPLOYMENT.md) for details.

### Improving Results

If model quality isn't satisfactory:

1. **More data**: 10K+ samples recommended
2. **Better data**: Higher quality examples
3. **More epochs**: 5-10 epochs for better learning
4. **Larger model**: Try 14B if using Pro+ (A100)
5. **Higher LoRA rank**: 32 or 64 for more capacity

---

## Cost Breakdown

### Colab Free
- **GPU**: Tesla T4 (15GB VRAM)
- **Session**: 12 hours maximum
- **Cost**: $0
- **Best for**: Learning, testing, small datasets

### Colab Pro ($10/month)
- **GPU**: T4/V100 (15-16GB VRAM)
- **Session**: 24 hours maximum
- **Priority**: Higher priority access
- **Best for**: Regular use, medium datasets

### Colab Pro+ ($50/month)
- **GPU**: V100/A100 (40GB VRAM)
- **Session**: 24 hours maximum
- **Priority**: Highest priority
- **Background**: Runs even when browser closed
- **Best for**: Larger models (14B), production work

### Cost Comparison

| Task | Free | Pro | Pro+ | RunPod |
|------|------|-----|------|--------|
| 1K samples, 7B | $0 | $0 | $0 | ~$1 |
| 10K samples, 7B | $0* | $0 | $0 | ~$5 |
| 10K samples, 14B | N/A | N/A | $0 | ~$8 |

*May require multiple sessions

---

## Summary Checklist

Before you finish:

- [ ] Model trained successfully
- [ ] Model saved to Google Drive
- [ ] Model tested with prompts
- [ ] Model downloaded or uploaded to HF
- [ ] Training configs saved
- [ ] Notebook saved to Drive (optional)

**Congratulations!** You've successfully fine-tuned an LLM on Google Colab! 🎉

## Quick Reference

```bash
# Essential commands
!nvidia-smi                                          # Check GPU
!python -m finetune_project prepare-data --config config.yaml  # Prepare data
!python -m finetune_project train --config config.yaml         # Train
!cp -r output /content/drive/MyDrive/llm-finetune/            # Save to Drive
```

## Support

- **Documentation**: [docs/README.md](README.md)
- **Full Colab Guide**: [COLAB_GUIDE.md](COLAB_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- **Notebook**: [llm_finetune_colab.ipynb](../notebooks/llm_finetune_colab.ipynb)
