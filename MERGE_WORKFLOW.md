# Complete Model Merge Workflow - Step by Step

This guide walks you through the complete process of merging your trained LoRA adapter with the base model on your local Windows machine.

## Prerequisites Checklist

Before you start, make sure you have:

- [ ] Completed training on RunPod
- [ ] Python 3.8+ installed on your local machine
- [ ] At least 30-60 GB free disk space on C: drive
- [ ] Required Python packages installed (see below)

## Step 1: Install Required Packages

Open Command Prompt or PowerShell and run:

```cmd
pip install transformers peft torch accelerate safetensors
```

**Fix if you get numpy/pandas error:**
```cmd
pip install --upgrade --force-reinstall pandas
```

## Step 2: Download Your Trained Model from RunPod

> 📥 **Detailed download instructions:** See [DOWNLOAD_FROM_RUNPOD.md](DOWNLOAD_FROM_RUNPOD.md) for multiple download methods

### Quick Download (Web Interface)

1. Log into [RunPod](https://www.runpod.io/)
2. Connect to your pod → "Connect to HTTP Service"
3. Navigate to `/workspace/llm-finetune/output/`
4. Download the `output` folder as ZIP
5. Extract to `c:\LLM\llm-finetune\output\` on your local machine

### Alternative: SCP (Command Line)

```cmd
# Replace POD_IP with your RunPod IP address
scp -r root@POD_IP:/workspace/llm-finetune/output c:\LLM\llm-finetune\
```

**What you should have downloaded:**
- `adapter_config.json` ✅
- `adapter_model.safetensors` (or `adapter_model.bin`) ✅
- `tokenizer.json`
- `tokenizer_config.json`
- Other tokenizer files

**Download size:** ~300-600 MB total

## Step 3: Organize Your Files

Place the downloaded `output` folder in the correct location:

```
c:\LLM\llm-finetune\
└── output\
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer files...
    └── ...
```

**Verify it's in the right place:**
```cmd
dir c:\LLM\llm-finetune\output
```

You should see `adapter_config.json` and `adapter_model.safetensors`.

## Step 4: Run the Merge Script

### Option A: Double-Click Method (Easiest)

1. Open File Explorer
2. Navigate to `c:\LLM\llm-finetune\`
3. Double-click `merge.bat`
4. Follow the on-screen prompts

### Option B: Command Line Method

```cmd
cd c:\LLM\llm-finetune
python scripts/merge_local.py
```

## Step 5: What Happens During Merge

The script will automatically:

1. **Find your adapter** at `c:\LLM\llm-finetune\output\`
   ```
   ✅ Found adapter at: c:\LLM\llm-finetune\output
   ```

2. **Detect base model** (7B or 14B)
   ```
   ✅ Detected base model: Qwen/Qwen2.5-7B-Instruct
   ```

3. **Check disk space** (needs 30-60 GB)
   ```
   Available space: 120.5 GB
   ✅ Sufficient disk space available
   ```

4. **Download base model** from HuggingFace
   ```
   Cache: c:\LLM\merged_model\base_model_cache
   ⏳ Downloading... (this takes 5-15 minutes)
   ✅ Base model loaded successfully
   ```

5. **Merge the models**
   ```
   ⏳ Merging adapter with base model...
   ✅ Model merged successfully
   ```

6. **Save merged model**
   ```
   📁 Output: c:\LLM\merged_model
   ✅ Model saved successfully
   ✅ Tokenizer saved successfully
   ```

**Total time:** 15-30 minutes (depending on internet speed and GPU/CPU)

## Step 6: Verify the Merge

After the merge completes, verify your files:

```cmd
dir c:\LLM\merged_model
```

You should see:
- `config.json`
- `generation_config.json`
- `model-00001-of-*.safetensors` (multiple files)
- `model.safetensors.index.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `base_model_cache\` (folder - can delete after merge)

**Check total size:**
```cmd
# Should be ~15-20 GB for 7B model, ~30-35 GB for 14B model
```

## Step 7: Test Your Merged Model

Run a quick test:

```cmd
cd c:\LLM\llm-finetune
python scripts/test_merged_model.py c:\LLM\merged_model
```

Or test with a custom prompt:

```cmd
python scripts/test_merged_model.py c:\LLM\merged_model --prompt "What is machine learning?"
```

**What to expect:**
- Model loads successfully
- Generates coherent responses
- No errors or warnings

## Step 8: Clean Up (Optional)

After verifying the merge worked, you can free up space:

```cmd
# Delete the base model cache (saves 15-30 GB)
rmdir /s /q c:\LLM\merged_model\base_model_cache
```

**What to keep:**
- ✅ `c:\LLM\merged_model\` - Your final merged model (KEEP THIS!)
- ✅ `c:\LLM\llm-finetune\output\` - Your adapter backup (small, ~500 MB)

**What you can delete:**
- ❌ `c:\LLM\merged_model\base_model_cache\` - No longer needed after merge

## Step 9: Use Your Merged Model

Now you can use your model in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your merged model
model = AutoModelForCausalLM.from_pretrained(
    "c:/LLM/merged_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("c:/LLM/merged_model")

# Use it
messages = [{"role": "user", "content": "Hello! How can you help me?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Final Directory Structure

After everything is complete:

```
c:\LLM\
├── llm-finetune\              # Project source code
│   ├── scripts\
│   ├── src\
│   ├── configs\
│   ├── merge.bat              # Merge helper script
│   └── output\                # Downloaded adapter (backup)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
│
└── merged_model\              # Your final merged model
    ├── config.json            # Model configuration
    ├── generation_config.json # Generation settings
    ├── model-00001-of-*.safetensors  # Model weights
    ├── model.safetensors.index.json
    ├── tokenizer.json         # Tokenizer
    ├── tokenizer_config.json
    ├── vocab.json
    └── merges.txt
```

## Troubleshooting

### Issue: "No adapter files found"

**Solution:**
```cmd
# Check if adapter exists
dir c:\LLM\llm-finetune\output\adapter_config.json

# If not, re-download from RunPod to correct location
```

### Issue: "Out of disk space"

**Solution:**
```cmd
# Check free space
dir c:\

# Free up space:
# 1. Delete temporary files
# 2. Empty recycle bin
# 3. Use a different drive with --output_path
python scripts/merge_local.py --output_path d:\merged_model
```

### Issue: "Out of memory" during merge

**Solution:**
```cmd
# Use CPU instead of GPU
python scripts/merge_local.py --device cpu

# Close other programs to free RAM
```

### Issue: numpy/pandas compatibility error

**Solution:**
```cmd
pip install --upgrade --force-reinstall pandas
```

### Issue: Model not generating properly

**Checklist:**
1. Verify merge completed successfully (no errors)
2. Check all model files are present
3. Test with simple prompt first
4. Verify base model matches training (7B vs 14B)

**Test adapter directly to confirm training worked:**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "c:/LLM/llm-finetune/output")
# If this works but merged doesn't, re-run merge
```

## Next Steps

After successfully merging:

1. **Test thoroughly** with various prompts
2. **Backup your model** to external drive or cloud
3. **Share or deploy** your model:
   - Upload to HuggingFace Hub
   - Use with inference servers (vLLM, TGI)
   - Create an API endpoint
   - Build an application

## Advanced Options

### Merge with Custom Paths

```cmd
python scripts/merge_local.py ^
  --adapter_path c:\Downloads\my_adapter ^
  --output_path d:\models\my_merged_model ^
  --base_model Qwen/Qwen2.5-7B-Instruct ^
  --device cuda
```

### Upload to HuggingFace Hub

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("c:/LLM/merged_model")
tokenizer = AutoTokenizer.from_pretrained("c:/LLM/merged_model")

# Login first: huggingface-cli login
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

## Quick Reference

```cmd
# Install packages
pip install transformers peft torch accelerate safetensors

# Run merge (simple)
cd c:\LLM\llm-finetune
python scripts/merge_local.py

# Test merged model
python scripts/test_merged_model.py c:\LLM\merged_model

# Clean up cache
rmdir /s /q c:\LLM\merged_model\base_model_cache
```

## Need Help?

- **Full Documentation:** [MERGE_GUIDE.md](MERGE_GUIDE.md)
- **Windows-Specific Guide:** [MERGE_LOCAL_WINDOWS.md](MERGE_LOCAL_WINDOWS.md)
- **GitHub Issues:** [Report a problem](https://github.com/ravidsun/llm-finetune/issues)
- **Main README:** [README.md](README.md)

---

## Summary Checklist

- [ ] Install Python packages
- [ ] Download `output/` folder from RunPod
- [ ] Place in `c:\LLM\llm-finetune\output\`
- [ ] Run `python scripts/merge_local.py`
- [ ] Wait for merge to complete (15-30 min)
- [ ] Verify files in `c:\LLM\merged_model\`
- [ ] Test model works
- [ ] Delete `base_model_cache/` to save space
- [ ] Use your model!

**You're done!** 🎉
