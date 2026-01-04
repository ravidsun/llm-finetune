# Merge Model Locally on Windows

Quick reference guide for merging your LoRA adapter with the base model on your local Windows machine.

> 📖 **For detailed step-by-step instructions, see [MERGE_WORKFLOW.md](MERGE_WORKFLOW.md)**

## Prerequisites

1. **Python installed** (3.8 or later)
2. **Required packages:**
   ```cmd
   pip install transformers peft torch accelerate safetensors

   # If you get numpy/pandas errors:
   pip install --upgrade --force-reinstall pandas
   ```

3. **Disk space:** 30-60 GB free (depending on model size)
4. **Downloaded adapter:** Get `output/` folder from RunPod to `c:\LLM\llm-finetune\output\`

## Quick Start (TL;DR)

### Option 1: Double-Click Method (Easiest) ⭐

1. Download `output/` folder from RunPod
2. Place it at `c:\LLM\llm-finetune\output\`
3. Double-click `merge.bat` in `c:\LLM\llm-finetune\`
4. Wait 15-30 minutes
5. Find merged model in `c:\LLM\merged_model\`

### Option 2: Command Line

```cmd
cd c:\LLM\llm-finetune
python scripts\merge_local.py
```

**Result:** Merged model will be in `c:\LLM\merged_model\`

## What Happens

The script will:
1. ✅ Auto-find adapter in `c:\LLM\llm-finetune\output\` (from your RunPod download)
2. ✅ Auto-detect which base model you used (7B or 14B)
3. ✅ Check disk space
4. ✅ Download base model from HuggingFace to `c:\LLM\merged_model\base_model_cache\`
5. ✅ Merge adapter with base model
6. ✅ Save final merged model to `c:\LLM\merged_model\`

Everything happens in the `c:\LLM\merged_model\` directory - completely separate from your project files!

## Advanced Usage

### Specify Custom Output Location

```cmd
# Save to a different location
python scripts\merge_local.py --output_path c:\MyModels\my_merged_model
```

### Specify Custom Adapter Path

```cmd
# If adapter is in a different location
python scripts\merge_local.py --adapter_path c:\Downloads\model_output
```

### Force Specific Base Model

```cmd
python scripts\merge_local.py --base_model Qwen/Qwen2.5-7B-Instruct
```

### Use CPU Only (if GPU memory issues)

```cmd
python scripts\merge_local.py --device cpu
```

### All Options Combined

```cmd
python scripts\merge_local.py ^
  --adapter_path c:\LLM\llm-finetune\output ^
  --output_path c:\LLM\merged_model ^
  --base_model Qwen/Qwen2.5-7B-Instruct ^
  --device cuda
```

## Testing Your Merged Model

```cmd
python scripts\test_merged_model.py c:\LLM\merged_model
```

Or with a custom prompt:

```cmd
python scripts\test_merged_model.py c:\LLM\merged_model --prompt "Your test question"
```

## Using the Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "c:/LLM/merged_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("c:/LLM/merged_model")

# Use it
messages = [{"role": "user", "content": "Hello!"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Default Paths

### Inputs (Read Only)
- **Adapter:** Auto-detected from:
  - `c:\LLM\llm-finetune\output\` (primary - where you download from RunPod)
  - `.\output\` (current directory fallback)

### Outputs (All in one place)
- **Merged Model:** `c:\LLM\merged_model\` (main output)
- **Base Model Cache:** `c:\LLM\merged_model\base_model_cache\` (downloaded from HuggingFace)

**Result Structure:**
```
c:\LLM\
├── llm-finetune\          # Your project (source code only)
│   └── output\            # Downloaded adapter from RunPod
└── merged_model\          # Everything merge-related
    ├── config.json        # Final merged model files
    ├── model-*.safetensors
    ├── tokenizer files...
    └── base_model_cache\  # Base model downloaded here
```

## Troubleshooting

### "No adapter files found"
- Make sure you have the `output/` folder from training
- Check that it's at `c:\LLM\llm-finetune\output\`
- Verify `adapter_model.safetensors` exists inside

### "Out of disk space"
- You need 30-60 GB free depending on model size
- Clean up other files or use a different drive

### "Out of memory"
- Use `--device cpu` to merge on CPU instead of GPU
- Close other programs to free up memory

### "Can't download base model"
- Check internet connection
- Verify base model name is correct (Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen2.5-14B-Instruct)

## File Sizes

- **7B Model:** ~15-20 GB
- **14B Model:** ~30-35 GB

## Next Steps

After merging:
1. Test the model works correctly
2. Use it for inference from `c:\LLM\merged_model\`
3. Optionally upload to HuggingFace Hub
4. Clean up if needed:
   - Delete `c:\LLM\merged_model\base_model_cache\` (saves ~15-30 GB)
   - Keep `c:\LLM\llm-finetune\output\` (your adapter - only ~500 MB)

## Cleaning Up

After successful merge, you can free up space:

```cmd
# Delete base model cache (safe - keeps your merged model)
rmdir /s /q c:\LLM\merged_model\base_model_cache

# This leaves only your final merged model in c:\LLM\merged_model\
```

**What to keep:**
- ✅ `c:\LLM\merged_model\` (final merged model - **keep this**)
- ✅ `c:\LLM\llm-finetune\output\` (your adapter - small, good backup)

**What to delete (optional):**
- ❌ `c:\LLM\merged_model\base_model_cache\` (can be deleted after merge)

## Complete Documentation

For more detailed information, see [MERGE_GUIDE.md](MERGE_GUIDE.md)

## Need Help?

- [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- [Full Documentation](docs/README.md)
