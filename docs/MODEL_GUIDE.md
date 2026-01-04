# Model Selection Guide for RunPod Fine-Tuning

Complete guide to choosing and configuring LLMs for fine-tuning on RunPod.

## Quick Recommendations

### 💰 Best Budget: Qwen/Qwen2.5-7B-Instruct
- **GPU**: RTX 4090 (24GB)
- **Cost**: $0.44/hr
- **10K samples**: $1.76-$3.52 (4-8 hours)

### ⚡ Best Performance/Value: Qwen/Qwen2.5-14B-Instruct ⭐ **CURRENT DEFAULT**
- **GPU**: A40 (46GB)
- **Cost**: $0.39/hr
- **10K samples**: $3.12-$5.46 (8-14 hours)

### 🏆 Maximum Quality: Meta-Llama-3.1-70B-Instruct
- **GPU**: A100 80GB
- **Cost**: $2.89/hr
- **10K samples**: $69-$116 (24-40 hours)

---

## Detailed Model Comparison

### 7B-8B Models (Entry Level)

#### 1. Qwen/Qwen2.5-7B-Instruct
```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  learning_rate: 2.0e-4
```

**Specs:**
- Parameters: 7 billion
- Context: 32K tokens
- Languages: 29+ languages (excellent multilingual)
- Strengths: Reasoning, coding, multilingual

**RunPod Setup:**
- **Best GPU**: RTX 4090 (24GB VRAM)
- **Cost**: ~$0.44/hour
- **Availability**: ⭐⭐⭐⭐⭐ Excellent

**Training Time & Cost:**
| Dataset Size | Time | Cost (RTX 4090) |
|--------------|------|-----------------|
| 1K samples   | 30-60 min | $0.22-0.44 |
| 5K samples   | 2-4 hours | $0.88-1.76 |
| 10K samples  | 4-8 hours | $1.76-3.52 |
| 50K samples  | 12-20 hours | $5.28-8.80 |

**Use Cases:**
- ✅ General instruction following
- ✅ Multilingual applications
- ✅ Budget-conscious projects
- ✅ Quick experimentation
- ✅ Coding assistance

---

#### 2. Meta-Llama-3.1-8B-Instruct
```yaml
model:
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  learning_rate: 2.0e-4
```

**Specs:**
- Parameters: 8 billion
- Context: 128K tokens (exceptional)
- Languages: Primarily English
- Strengths: Long context, reasoning, general tasks

**RunPod Setup:**
- **Best GPU**: RTX 4090 (24GB VRAM)
- **Cost**: ~$0.44/hour
- **Availability**: ⭐⭐⭐⭐⭐ Excellent

**Training Time & Cost:**
| Dataset Size | Time | Cost (RTX 4090) |
|--------------|------|-----------------|
| 1K samples   | 30-70 min | $0.22-0.51 |
| 5K samples   | 2.5-4.5 hours | $1.10-1.98 |
| 10K samples  | 5-9 hours | $2.20-3.96 |

**Use Cases:**
- ✅ Long-context applications (128K!)
- ✅ English-focused tasks
- ✅ Strong reasoning requirements
- ✅ Chat applications

---

#### 3. mistralai/Mistral-7B-Instruct-v0.3
```yaml
model:
  model_name: "mistralai/Mistral-7B-Instruct-v0.3"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  learning_rate: 2.0e-4
```

**Specs:**
- Parameters: 7 billion
- Context: 32K tokens
- Languages: English, some multilingual
- Strengths: Speed, efficiency, general purpose

**RunPod Setup:**
- **Best GPU**: RTX 4090 (24GB VRAM)
- **Cost**: ~$0.44/hour
- **Availability**: ⭐⭐⭐⭐⭐ Excellent

**Training Time & Cost:**
| Dataset Size | Time | Cost (RTX 4090) |
|--------------|------|-----------------|
| 1K samples   | 25-50 min | $0.18-0.37 |
| 5K samples   | 2-3.5 hours | $0.88-1.54 |
| 10K samples  | 4-7 hours | $1.76-3.08 |

**Use Cases:**
- ✅ Speed-critical applications
- ✅ Lower latency requirements
- ✅ General purpose tasks
- ✅ Cost optimization

---

### 14B-20B Models (Mid-Range)

#### 4. Qwen/Qwen2.5-14B-Instruct ⭐ **RECOMMENDED**
```yaml
model:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  lora_rank: 32
  lora_alpha: 64
  use_qlora: true

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 4096
  learning_rate: 1.5e-4
```

**Specs:**
- Parameters: 14 billion
- Context: 32K tokens
- Languages: 29+ languages
- Strengths: Best performance/price, multilingual, reasoning

**RunPod Setup:**
- **Best GPU**: A40 (46GB VRAM) ⭐ **BEST VALUE**
- **Alternative**: A100 40GB (faster but pricier)
- **Cost**: $0.39/hr (A40) or $1.89/hr (A100)
- **Availability**: ⭐⭐⭐⭐ Good

**Training Time & Cost:**
| Dataset Size | Time (A40) | Cost (A40) | Time (A100) | Cost (A100) |
|--------------|------------|------------|-------------|-------------|
| 1K samples   | 1-2 hours  | $0.39-0.78 | 40-60 min   | $1.26-1.89  |
| 5K samples   | 4-7 hours  | $1.56-2.73 | 2-4 hours   | $3.78-7.56  |
| 10K samples  | 8-14 hours | $3.12-5.46 | 4-8 hours   | $7.56-15.12 |
| 50K samples  | 24-36 hours| $9.36-14.04| 12-20 hours | $22.68-37.80|

**Use Cases:**
- ✅ Professional applications
- ✅ Complex reasoning tasks
- ✅ Multilingual requirements
- ✅ Domain expertise (medical, legal, etc.)
- ✅ Best quality/cost balance

**Why A40 over A100?**
- A40 is 5x cheaper ($0.39 vs $1.89/hr)
- Only 20-30% slower for fine-tuning
- 46GB VRAM is plenty for 14B with QLoRA
- Better availability

---

#### 5. Meta-Llama-3.1-70B-Instruct (Premium)
```yaml
model:
  model_name: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  lora_rank: 64
  lora_alpha: 128
  use_qlora: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 4096
  learning_rate: 1.0e-4
```

**Specs:**
- Parameters: 70 billion
- Context: 128K tokens
- Languages: English (some multilingual)
- Strengths: State-of-the-art performance

**RunPod Setup:**
- **Required GPU**: A100 80GB
- **Cost**: ~$2.89/hour
- **Availability**: ⭐⭐ Limited

**Training Time & Cost:**
| Dataset Size | Time | Cost (A100 80GB) |
|--------------|------|------------------|
| 1K samples   | 3-5 hours  | $8.67-14.45  |
| 5K samples   | 12-20 hours| $34.68-57.80 |
| 10K samples  | 24-40 hours| $69.36-115.60|

**Use Cases:**
- ✅ Maximum quality requirements
- ✅ Highly specialized domains
- ✅ Research projects
- ✅ When cost is not primary concern
- ❌ Not recommended for learning/testing

---

## Specialized Models

### Code Models

#### deepseek-ai/deepseek-coder-33b-instruct
```yaml
model:
  model_name: "deepseek-ai/deepseek-coder-33b-instruct"
  lora_rank: 32
  lora_alpha: 64
  use_qlora: true
```

- **GPU**: A100 40GB ($1.89/hr)
- **Strengths**: Code generation, debugging, 80+ languages
- **Context**: 16K tokens
- **Use**: Code-specific fine-tuning

#### codellama/CodeLlama-34b-Instruct-hf
```yaml
model:
  model_name: "codellama/CodeLlama-34b-Instruct-hf"
  lora_rank: 32
  lora_alpha: 64
  use_qlora: true
```

- **GPU**: A100 40GB ($1.89/hr)
- **Strengths**: Code completion, infilling
- **Context**: 16K tokens
- **Use**: Programming assistance

---

### Multilingual Models

#### CohereForAI/aya-23-8B
```yaml
model:
  model_name: "CohereForAI/aya-23-8B"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true
```

- **GPU**: RTX 4090 ($0.44/hr)
- **Languages**: 23 languages
- **Strengths**: Diverse language support
- **Use**: Multilingual applications

---

### Function Calling

#### NousResearch/Hermes-2-Pro-Llama-3-8B
```yaml
model:
  model_name: "NousResearch/Hermes-2-Pro-Llama-3-8B"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true
```

- **GPU**: RTX 4090 ($0.44/hr)
- **Strengths**: Function calling, tool use
- **Use**: Agents, API integration

---

## GPU Selection Guide

### GPU Comparison

| GPU | VRAM | Cost/Hr | Best For | Availability |
|-----|------|---------|----------|--------------|
| RTX A4000 | 16GB | $0.24 | 7B (tight) | ⭐⭐⭐ |
| **RTX 4090** | 24GB | $0.44 | **7B-8B** ⭐ | ⭐⭐⭐⭐⭐ |
| **A40** | 46GB | $0.39 | **14B-20B** ⭐ | ⭐⭐⭐⭐ |
| A100 40GB | 40GB | $1.89 | 14B-30B | ⭐⭐⭐ |
| A100 80GB | 80GB | $2.89 | 70B+ | ⭐⭐ |
| RTX 6000 Ada | 48GB | $0.79 | 14B-20B | ⭐⭐⭐⭐ |

### GPU Decision Tree

```
How many parameters?
├─ 7B-8B → RTX 4090 ($0.44/hr)
├─ 14B-20B → A40 ($0.39/hr) ⭐ BEST VALUE
├─ 30B-34B → A100 40GB ($1.89/hr)
└─ 70B+ → A100 80GB ($2.89/hr)
```

---

## Configuration Parameters Explained

### LoRA Settings

```yaml
lora_rank: 32           # Higher = more capacity (16 for 7B, 32 for 14B, 64 for 70B)
lora_alpha: 64          # Usually 2x lora_rank
lora_dropout: 0.05      # Regularization (0.05-0.1)
use_qlora: true         # 4-bit quantization (essential for large models)
```

**LoRA Rank Guidelines:**
- 7B models: 8-16
- 14B models: 16-32
- 30B models: 32-64
- 70B models: 64-128

### Batch Settings

```yaml
per_device_train_batch_size: 4    # Samples per GPU pass
gradient_accumulation_steps: 4     # Accumulate before update
# Effective batch size = 4 × 4 = 16
```

**GPU-Based Guidelines:**
- RTX 4090 (24GB): batch_size=2, grad_accum=8
- A40 (46GB): batch_size=4, grad_accum=4
- A100 40GB: batch_size=4-8, grad_accum=2-4

### Learning Rate

```yaml
learning_rate: 1.5e-4   # Lower for larger models
warmup_ratio: 0.03      # 3% warmup
lr_scheduler_type: "cosine"
```

**Model-Based Guidelines:**
- 7B models: 2.0e-4
- 14B models: 1.5e-4
- 30B+ models: 1.0e-4

### Sequence Length

```yaml
max_seq_length: 4096    # Maximum token length
```

**Trade-offs:**
- Longer = better context, slower training, more VRAM
- Shorter = faster training, less VRAM
- Sweet spot: 2048-4096 for most tasks

---

## Cost Optimization Tips

### 1. Use Spot Instances
- **Savings**: 50-70% cheaper
- **Risk**: Can be interrupted
- **Best for**: Datasets with frequent checkpoints

### 2. Start Small
```bash
# Test with 100-1000 samples first
# Validates config before full training
# Cost: $0.10-0.50
```

### 3. Monitor Training Loss
```bash
# Stop early if loss plateaus
# Can save 20-40% of training time
tail -f output/training.log
```

### 4. Choose Right GPU
- Don't overpay for A100 if A40 works
- A40 is 5x cheaper for 14B models
- RTX 4090 perfect for 7B models

### 5. Batch Size Optimization
```yaml
# Use maximum batch size your GPU allows
# Reduces training time by 20-30%
# Our script auto-optimizes this
```

---

## Quick Start Commands

### Update Existing Config to 14B Model

```bash
cd /workspace/llm-finetune

# Edit config
nano configs/existing_jsonl.yaml

# Change model_name to:
# model_name: "Qwen/Qwen2.5-14B-Instruct"
# lora_rank: 32
# lora_alpha: 64

# Train
bash scripts/runpod.sh train
```

### Auto-Generated Config (RunPod Script)

The `scripts/runpod.sh` script automatically detects GPU and sets optimal parameters:

```bash
# A40 (46GB) → Qwen2.5-14B-Instruct, batch_size=4
# RTX 4090 (24GB) → Qwen2.5-7B-Instruct, batch_size=2
# A100 80GB → Qwen2.5-14B-Instruct, batch_size=8
```

---

## FAQs

### Q: Should I use Qwen or Llama?
**A**:
- **Qwen**: Better multilingual, slightly better reasoning, 29+ languages
- **Llama 3.1**: Better long context (128K), strong English performance
- **Winner**: Qwen for most use cases, Llama if you need >32K context

### Q: Is 14B worth the extra cost over 7B?
**A**: Yes, if:
- ✅ You need higher quality outputs
- ✅ Your domain is complex (medical, legal, technical)
- ✅ You're fine-tuning for production use
- ❌ No, if just experimenting or learning

### Q: Can I use models not listed here?
**A**: Yes! Any Hugging Face model works. Just:
1. Check model size vs GPU VRAM
2. Adjust LoRA rank appropriately
3. Test with small dataset first

### Q: How do I know if OOM (out of memory)?
**A**: Look for:
```
CUDA out of memory
RuntimeError: CUDA error
```
**Fix**: Reduce batch_size or max_seq_length

### Q: What's the minimum dataset size?
**A**:
- Minimum: 100 samples (will overfit)
- Recommended: 1,000+ samples
- Good: 5,000-10,000 samples
- Excellent: 50,000+ samples

---

## Model Migration Guide

### From 7B to 14B

```bash
# 1. Update config
sed -i 's/Qwen2.5-7B/Qwen2.5-14B/' config.yaml
sed -i 's/lora_rank: 16/lora_rank: 32/' config.yaml
sed -i 's/lora_alpha: 32/lora_alpha: 64/' config.yaml
sed -i 's/max_seq_length: 2048/max_seq_length: 4096/' config.yaml

# 2. Change RunPod GPU from RTX 4090 to A40

# 3. Retrain
bash scripts/runpod.sh train
```

### From OpenAI GPT to Open Source

| OpenAI Model | Open Source Equivalent | GPU Needed |
|--------------|------------------------|------------|
| GPT-3.5 Turbo | Qwen2.5-7B-Instruct | RTX 4090 |
| GPT-4 | Qwen2.5-14B-Instruct | A40 |
| GPT-4 Turbo | Llama-3.1-70B-Instruct | A100 80GB |

---

## Support

- **Issues**: [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- **Docs**: [Complete Documentation](README.md)
- **RunPod Guide**: [RUNPOD_GUIDE.md](../RUNPOD_GUIDE.md)
