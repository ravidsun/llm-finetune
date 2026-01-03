# Phase 4: Model Evaluation & Export

This guide covers evaluating your fine-tuned model and exporting it for deployment.

## Prerequisites

✅ Completed [PHASE3_TRAINING.md](PHASE3_TRAINING.md)
✅ Training outputs in `/workspace/output/`

## Step 1: Quick Test the LoRA Adapter

Test your trained model with a simple prompt:

```python
# Create test script
cat > /workspace/test_model.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Your base model
adapter_path = "/workspace/output"             # LoRA adapter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Test prompt
prompt = "Explain quantum computing in simple terms:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
EOF

# Run test
python /workspace/test_model.py
```

## Step 2: Evaluate on Test Set (Optional)

If you have a test dataset:

```bash
# Prepare test data in same format as training data
cp test_data.jsonl /workspace/data/test.jsonl

# Run evaluation
python -m finetune_project evaluate \
    --config /workspace/configs/my_config.yaml \
    --test-file /workspace/data/test.jsonl
```

This computes:
- Perplexity
- Loss
- Token-level accuracy

## Step 3: Interactive Testing

Create an interactive chat script:

```python
cat > /workspace/chat.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

def chat(model, tokenizer):
    print("Chat with your fine-tuned model (type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        # Format prompt (adjust based on your prompt template)
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response = response.split("### Response:")[-1].strip()

        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    adapter = "/workspace/output"

    print("Loading model...")
    model, tokenizer = load_model(base_model, adapter)
    print("Model loaded!")

    chat(model, tokenizer)
EOF

# Run interactive chat
python /workspace/chat.py
```

## Step 4: Compare Base vs Fine-tuned

Create a comparison script:

```python
cat > /workspace/compare_models.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load base model
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load fine-tuned model
print("Loading fine-tuned model...")
ft_model = PeftModel.from_pretrained(base_model, "/workspace/output")

# Test prompts
prompts = [
    "Explain machine learning:",
    "What is deep learning?",
    "How does neural network work?"
]

for prompt in prompts:
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)

    print("\n[BASE MODEL]")
    base_response = generate_response(base_model, tokenizer, prompt)
    print(base_response)

    print("\n[FINE-TUNED MODEL]")
    ft_response = generate_response(ft_model, tokenizer, prompt)
    print(ft_response)
EOF

python /workspace/compare_models.py
```

## Step 5: Merge LoRA Adapter with Base Model

Merging creates a standalone model without needing PEFT:

```bash
python -m finetune_project merge-adapter \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter /workspace/output \
    --output /workspace/merged-model
```

This creates a full model at `/workspace/merged-model/`.

**Merged model benefits**:
- ✅ Faster inference (no adapter overhead)
- ✅ Compatible with more inference frameworks
- ✅ Easier to share/deploy

**Merged model size**:
- 7B model: ~14-15 GB
- 14B model: ~28-30 GB
- 32B model: ~64-65 GB

## Step 6: Export for Deployment

### Option A: Hugging Face Format (Default)

Already done! Use `/workspace/merged-model/` or `/workspace/output/` (adapter).

Upload to Hugging Face Hub:
```bash
# Login to HF
huggingface-cli login

# Upload merged model
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/workspace/merged-model',
    repo_id='your-username/your-model-name',
    repo_type='model'
)
"
```

### Option B: GGUF Format (for llama.cpp)

Convert to GGUF for CPU/local inference:

```bash
# Install llama.cpp tools
pip install gguf

# Convert to GGUF
python -m finetune_project export \
    --model /workspace/merged-model \
    --output /workspace/export \
    --format gguf

# Quantize (optional, reduces size)
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Quantize to different levels
./quantize /workspace/export/model.gguf /workspace/export/model-q4.gguf Q4_K_M
./quantize /workspace/export/model.gguf /workspace/export/model-q8.gguf Q8_0
```

Quantization levels:
- **Q4_K_M**: 4-bit, ~3.5-4 GB (7B model), good quality
- **Q5_K_M**: 5-bit, ~4.5-5 GB, better quality
- **Q8_0**: 8-bit, ~7-8 GB, near original quality

### Option C: ONNX Format (for ONNX Runtime)

```bash
pip install optimum[exporters]

optimum-cli export onnx \
    --model /workspace/merged-model \
    --task text-generation \
    /workspace/export/onnx/
```

### Option D: vLLM Format (for high-throughput serving)

vLLM can directly use HuggingFace format:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/workspace/merged-model")
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
outputs = llm.generate(prompts, sampling_params)
```

## Step 7: Download Model to Local Machine

### Download via SCP:

```bash
# From local machine
scp -r root@<pod-ip>:/workspace/output ./my-lora-adapter
scp -r root@<pod-ip>:/workspace/merged-model ./my-full-model
```

### Download via RunPod Sync:

```bash
# On RunPod pod
cd /workspace
zip -r my-model.zip merged-model/

# Download via browser from RunPod file manager
```

### Upload to Cloud Storage:

```bash
# AWS S3
aws s3 cp /workspace/merged-model/ s3://my-bucket/my-model/ --recursive

# Google Cloud Storage
gsutil -m cp -r /workspace/merged-model/ gs://my-bucket/my-model/
```

## Step 8: Benchmark Performance

### Measure Inference Speed:

```python
cat > /workspace/benchmark.py <<'EOF'
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "/workspace/output"

# Load
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# Benchmark
prompt = "Explain quantum computing:" * 10  # Longer prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Warmup
model.generate(**inputs, max_new_tokens=100)

# Measure
num_runs = 10
total_time = 0
total_tokens = 0

for _ in range(num_runs):
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=256)
    end = time.time()

    total_time += (end - start)
    total_tokens += outputs.shape[1]

avg_time = total_time / num_runs
avg_tokens = total_tokens / num_runs
tokens_per_sec = avg_tokens / avg_time

print(f"Average generation time: {avg_time:.2f}s")
print(f"Average tokens: {avg_tokens:.0f}")
print(f"Tokens per second: {tokens_per_sec:.1f}")
EOF

python /workspace/benchmark.py
```

## Evaluation Checklist

- [ ] Model generates coherent responses
- [ ] Responses align with training domain
- [ ] No repetition or degeneration
- [ ] Performance meets requirements
- [ ] Model exported in desired format
- [ ] Artifacts backed up/uploaded

## Troubleshooting

### Model outputs gibberish

1. Check training loss - should be < 1.0
2. Verify data quality in training set
3. Try lower temperature: `temperature=0.3`
4. Check if wrong prompt template used

### Model too slow

1. Use merged model instead of adapter
2. Quantize to Q4/Q8 GGUF
3. Use vLLM or TensorRT-LLM for inference
4. Enable Flash Attention 2

### Export fails

```bash
# Ensure enough disk space
df -h /workspace

# Clear cache
rm -rf ~/.cache/huggingface/hub/*

# Try exporting adapter only (smaller)
python -m finetune_project export --model /workspace/output --output /workspace/export
```

## Next Steps

✅ Evaluation complete! Proceed to [PHASE5_DEPLOYMENT.md](PHASE5_DEPLOYMENT.md)

## Quick Reference

```bash
# Test model
python test_model.py

# Interactive chat
python chat.py

# Merge adapter
python -m finetune_project merge-adapter \
    --base-model MODEL_NAME \
    --adapter /workspace/output \
    --output /workspace/merged-model

# Export to GGUF
python -m finetune_project export \
    --model /workspace/merged-model \
    --output /workspace/export \
    --format gguf

# Download model
scp -r root@pod:/workspace/merged-model ./my-model
```
