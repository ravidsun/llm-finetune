# Merge Model Locally on Windows

Quick guide for merging your LoRA adapter with the base model on your local Windows machine.

## Prerequisites

1. **Python installed** (3.8 or later)
2. **Required packages:**
   ```cmd
   pip install transformers peft torch accelerate safetensors
   ```

3. **Disk space:** 30-60 GB free (depending on model size)

## Quick Start

### Option 1: Double-Click Method (Easiest)

1. Download your trained `output/` folder from RunPod to your local machine
2. Place it in the `llm-finetune/` directory
3. Double-click `merge.bat` in Windows Explorer
4. Follow the prompts

### Option 2: Command Line

```cmd
cd c:\LLM\llm-finetune
python scripts\merge_local.py
```

## What Happens

The script will:
1. ✅ Check for adapter files in `./output/`
2. ✅ Auto-detect which base model you used (7B or 14B)
3. ✅ Check disk space
4. ✅ Download base model from HuggingFace (if not cached)
5. ✅ Merge adapter with base model
6. ✅ Save merged model to `./merged_model/`

## Advanced Usage

### Specify Custom Paths

```cmd
python scripts\merge_local.py --adapter_path .\my_output --output_path .\my_merged
```

### Force Specific Base Model

```cmd
python scripts\merge_local.py --base_model Qwen/Qwen2.5-7B-Instruct
```

### Use CPU Only (if GPU memory issues)

```cmd
python scripts\merge_local.py --device cpu
```

## Testing Your Merged Model

```cmd
python scripts\test_merged_model.py .\merged_model
```

Or with a custom prompt:

```cmd
python scripts\test_merged_model.py .\merged_model --prompt "Your test question"
```

## Using the Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Use it
messages = [{"role": "user", "content": "Hello!"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### "No adapter files found"
- Make sure you have the `output/` folder from training
- Check that `output/adapter_model.safetensors` exists

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
2. Use it for inference
3. Optionally upload to HuggingFace Hub
4. Clean up the `output/` folder if you want to save space

## Complete Documentation

For more detailed information, see [MERGE_GUIDE.md](MERGE_GUIDE.md)

## Need Help?

- [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- [Full Documentation](docs/README.md)
