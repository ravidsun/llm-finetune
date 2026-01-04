# Using Existing JSONL Files

If you already have pre-processed JSONL training files, you can use them directly without data preparation.

## Standard Directory Structure

We recommend using `/workspace/data/input` as the standard location for your existing JSONL files:

**RunPod/Local:**
```
/workspace/
├── data/
│   └── input/              ← Place your JSONL files here
│       ├── train.jsonl
│       └── test.jsonl      (optional)
├── processed_data/         ← Processed output goes here
└── output/                 ← Training output (model weights)
```

**RunPod:**
```
/content/
├── data/
│   └── input/              ← Place your JSONL files here
│       ├── train.jsonl
│       └── test.jsonl      (optional)
├── processed_data/         ← Processed output goes here
└── output/                 ← Training output (model weights)
```

## Quick Start

### Option 1: Skip Data Preparation Entirely

If your JSONL file is already in the correct format, you can skip the preparation step and train directly.

#### Step 1: Place Your JSONL File

**RunPod/Local:**
```bash
# Create processed_data directory
mkdir -p /workspace/processed_data

# Copy your existing JSONL file
cp /path/to/your/file.jsonl /workspace/processed_data/train.jsonl
```

**RunPod:**
```bash
# Create processed_data directory
!mkdir -p processed_data

# Upload your file
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  # Point to where your JSONL already is
  # RunPod/Local: use /workspace/processed_data
  # RunPod: use processed_data (relative path)
  input_path: "/workspace/processed_data"  # or "processed_data" for RunPod
  output_path: "/workspace/processed_data"  # or "processed_data" for RunPod
  input_type: "json"

  # Disable LangChain processing since data is ready
  langchain:
    enabled: false

  augmentation:
    enabled: false

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "output"
```

#### Step 3: Train Directly

```bash
# Skip prepare-data step entirely
python -m finetune_project train --config config.yaml
```

---

## Option 2: Use Existing Files with Preparation

If you want to optionally apply augmentation or other processing to existing JSONL files:

### Step 1: Place Files in Input Directory

**RunPod/Local:**
```bash
# Create input directory
mkdir -p /workspace/data/input

# Copy your JSONL files
cp /path/to/your/train.jsonl /workspace/data/input/
cp /path/to/your/test.jsonl /workspace/data/input/  # Optional
```

**RunPod:**
```bash
# Create input directory
!mkdir -p data/input

# Upload your files
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  # Standard input directory for existing JSONL files
  # RunPod/Local: /workspace/data/input
  # RunPod: data/input (relative path)
  input_path: "/workspace/data/input"   # or "data/input" for RunPod
  output_path: "/workspace/processed_data"   # or "processed_data" for RunPod
  input_type: "json"                    # Important: use "json" for JSONL

  langchain:
    enabled: false  # Disable document processing

  augmentation:
    enabled: true   # Optional: enable to augment your existing data
    instruction_variations: true
    num_instruction_variations: 2

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "output"
```

### Step 3: Prepare (Optional) and Train

```bash
# If you want augmentation or validation
python -m finetune_project prepare-data --config config.yaml

# Or if you disabled augmentation, skip to training
python -m finetune_project train --config config.yaml
```

---

## JSONL Format Requirements

Your JSONL file should have one of these formats:

### Format 1: Instruction Format (Recommended)

```jsonl
{"instruction": "What is Python?", "input": "", "output": "Python is a programming language..."}
{"instruction": "Explain loops", "input": "in Python", "output": "Loops in Python allow..."}
```

**Fields**:
- `instruction`: The task or question
- `input`: Additional context (can be empty string)
- `output`: The expected response

### Format 2: Text Format (For Causal LM)

```jsonl
{"text": "This is a complete training example with all text in one field."}
{"text": "Another example. The model will learn to continue from any prefix."}
```

**Fields**:
- `text`: Complete training text

### Format 3: Conversation Format

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "Help me"}, {"role": "assistant", "content": "Sure!"}]}
```

**Fields**:
- `messages`: List of turn-taking messages

---

## Validation

Before training, validate your JSONL file:

```bash
# Check file format
python -m finetune_project validate --config config.yaml

# Or manually check
head -n 5 data/train.jsonl

# Count lines
wc -l data/train.jsonl
```

### Common Issues

**Issue**: `JSONDecodeError`
```bash
# Check for malformed JSON
python -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Error on line {i}: {e}')
"
```

**Issue**: Missing fields
```bash
# Check all entries have required fields
python -c "
import json
required = ['instruction', 'input', 'output']
with open('data/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        missing = [f for f in required if f not in data]
        if missing:
            print(f'Line {i} missing: {missing}')
"
```

---

## Directory Structure Examples

### Example 1: Direct Training (No Preparation)

```
your-project/
├── config.yaml
├── processed_data/
│   └── train.jsonl          # Your existing file (already formatted)
└── output/                  # Training output
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Workflow**:
```bash
# Place file
cp my-data.jsonl processed_data/train.jsonl

# Train directly
python -m finetune_project train --config config.yaml
```

---

### Example 2: With Optional Processing

```
your-project/
├── config.yaml
├── data/
│   ├── train.jsonl          # Your existing files
│   └── test.jsonl
├── processed_data/          # After preparation
│   ├── train.jsonl          # Processed/augmented
│   └── test.jsonl
└── output/                  # Training output
```

**Workflow**:
```bash
# Place files
cp my-train.jsonl data/train.jsonl
cp my-test.jsonl data/test.jsonl

# Optional: Process/augment
python -m finetune_project prepare-data --config config.yaml

# Train
python -m finetune_project train --config config.yaml
```

---

## Example Configurations

### Config 1: Pre-formatted JSONL (Skip All Processing)

```yaml
# config-preformatted.yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  use_qlora: true

data:
  input_path: "processed_data"    # Already has train.jsonl
  output_path: "processed_data"   # Same location
  input_type: "json"

  langchain:
    enabled: false

  augmentation:
    enabled: false

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  max_seq_length: 2048
  gradient_checkpointing: true
  bf16: true
  output_dir: "output"
  save_strategy: "epoch"
  logging_steps: 10
```

**Usage**:
```bash
# Your JSONL is already ready
cp my-data.jsonl processed_data/train.jsonl

# Skip prepare-data, go straight to training
python -m finetune_project train --config config-preformatted.yaml
```

---

### Config 2: Existing JSONL + Augmentation

```yaml
# config-augment.yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  input_path: "data"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false

  augmentation:
    enabled: true                       # Apply augmentation
    instruction_variations: true
    num_instruction_variations: 3       # Create 3 variations per sample
    use_paraphrase_templates: true
    whitespace_normalization: true

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "output"
```

**Usage**:
```bash
# Place your existing JSONL
cp my-data.jsonl data/train.jsonl

# Run preparation (applies augmentation)
python -m finetune_project prepare-data --config config-augment.yaml

# Train on augmented data
python -m finetune_project train --config config-augment.yaml
```

---

### Config 3: Multiple Existing Files

```yaml
# config-multiple.yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  input_path: "data"           # Contains multiple .jsonl files
  output_path: "processed_data"
  input_type: "json"

  # System will automatically combine all .jsonl files in data/

  langchain:
    enabled: false

  augmentation:
    enabled: false

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "output"
```

**Directory**:
```
data/
├── domain1.jsonl
├── domain2.jsonl
└── domain3.jsonl
```

**Usage**:
```bash
# All files will be combined
python -m finetune_project prepare-data --config config-multiple.yaml
python -m finetune_project train --config config-multiple.yaml
```

---

## RunPod Example

### Upload Existing JSONL to RunPod

```python
    !mv "{filename}" data/train.jsonl

print(f"✅ File uploaded: {list(uploaded.keys())[0]}")
!ls -lh data/
```

### Create Config for Direct Use

```python
config_yaml = """
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  use_qlora: true

data:
  input_path: "data"
  output_path: "processed_data"
  input_type: "json"

  langchain:
    enabled: false

  augmentation:
    enabled: false  # Set true if you want augmentation

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  max_seq_length: 512
  gradient_checkpointing: true
  output_dir: "output"
"""

with open('config.yaml', 'w') as f:
    f.write(config_yaml)
```

### Prepare and Train

```python
# Prepare data (validates and optionally augments)
!python -m finetune_project prepare-data --config config.yaml

# Train
!python -m finetune_project train --config config.yaml
```

---

## RunPod Example

### Upload Files via SCP

```bash
# From local machine
scp my-train.jsonl root@<pod-ip>:/workspace/llm-finetune/data/
scp my-test.jsonl root@<pod-ip>:/workspace/llm-finetune/data/
```

### Or Use Google Drive

```bash
# On RunPod
cd /workspace/llm-finetune

# If files are in Google Drive
# Mount drive first, then copy
cp /path/from/drive/*.jsonl data/
```

### Train

```bash
# Create config pointing to existing files
cat > config.yaml <<EOF
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  use_qlora: true

data:
  input_path: "data"
  output_path: "processed_data"
  input_type: "json"
  langchain:
    enabled: false
  augmentation:
    enabled: false

training:
  num_epochs: 3
  output_dir: "output"
EOF

# Prepare (validates format)
python -m finetune_project prepare-data --config config.yaml

# Train
python -m finetune_project train --config config.yaml
```

---

## FAQ

### Q: Can I use JSONL files that don't have instruction/input/output format?

**A**: Yes, if your JSONL has a `text` field:

```jsonl
{"text": "Complete training example here"}
```

Use this config:
```yaml
data:
  input_type: "text"  # Instead of "json"
```

### Q: Can I mix PDF files and JSONL files?

**A**: No, choose one input type per training run. If you have both:
1. Process PDFs separately to create JSONL
2. Combine all JSONL files
3. Train on combined JSONL

### Q: What if my JSONL has different field names?

**A**: Rename fields or create a conversion script:

```python
import json

# Convert your format to expected format
with open('your-data.jsonl') as f_in, open('converted.jsonl', 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        converted = {
            'instruction': data['question'],  # Your field name
            'input': data.get('context', ''),
            'output': data['answer']          # Your field name
        }
        f_out.write(json.dumps(converted) + '\n')
```

### Q: How do I validate my JSONL before training?

**A**: Use the validate command:

```bash
python -m finetune_project validate --config config.yaml
```

Or check manually:
```bash
# Check format
python -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            print(f'Line {i+1}: OK - {list(data.keys())}')
            if i >= 4:  # Show first 5
                break
        except Exception as e:
            print(f'Line {i+1}: ERROR - {e}')
"
```

### Q: Can I skip the prepare-data step entirely?

**A**: Yes, if your JSONL is already in the correct format:

```bash
# Place file in processed_data/
cp my-data.jsonl processed_data/train.jsonl

# Train directly
python -m finetune_project train --config config.yaml
```

Make sure your config has:
```yaml
data:
  input_path: "processed_data"
  output_path: "processed_data"
```

---

## Summary

### Quick Reference

| Scenario | Input Path | Prepare Data? | Config Setting |
|----------|-----------|---------------|----------------|
| Pre-formatted, ready to train | `processed_data/` | ❌ Skip | `input_path: "processed_data"` |
| Existing, want augmentation | `data/` | ✅ Run | `augmentation.enabled: true` |
| Existing, want validation | `data/` | ✅ Run | `augmentation.enabled: false` |
| Multiple JSONL files | `data/` | ✅ Run | All `.jsonl` combined automatically |

### Key Points

✅ **You can use existing JSONL files directly**
✅ **No need to convert from PDF/DOCX if you already have JSONL**
✅ **Place files in `data/` and set `input_type: "json"`**
✅ **Optionally enable augmentation to expand your dataset**
✅ **Can skip prepare-data if files are already formatted correctly**

Need help? Check the [main documentation](README.md) or open an issue on GitHub.
