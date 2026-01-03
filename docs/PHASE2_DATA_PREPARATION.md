# Phase 2: Data Preparation

This guide covers preparing your data for fine-tuning using LangChain document processing.

## Prerequisites

✅ Completed [PHASE1_SETUP.md](PHASE1_SETUP.md)

## Step 1: Choose Your Data Type

The pipeline supports three input types:

### Option A: JSON/JSONL (Recommended for instruction tuning)
### Option B: PDF Documents (For continued pretraining or QA generation)
### Option C: DOCX/TXT Documents (For continued pretraining or QA generation)

## Step 2: Prepare Your Data

### Option A: JSON Format

Place your JSONL file in `/workspace/data/`:

```jsonl
{"instruction": "What is the capital of France?", "input": "", "output": "Paris"}
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses quantum bits..."}
{"instruction": "Write a function to reverse a string", "input": "Python", "output": "def reverse_string(s):\n    return s[::-1]"}
```

Format requirements:
- One JSON object per line
- Fields: `instruction`, `input`, `output`
- `input` can be empty string if not needed

### Option B: PDF Documents

Upload PDF files to `/workspace/data/`:

```bash
# Example: Upload PDFs via SCP
scp my_documents/*.pdf root@<pod-ip>:/workspace/data/

# Or download from URL
cd /workspace/data
wget https://example.com/document.pdf
```

### Option C: DOCX/TXT Documents

Upload DOCX or TXT files to `/workspace/data/`:

```bash
# Upload files
scp my_documents/*.docx root@<pod-ip>:/workspace/data/
scp my_documents/*.txt root@<pod-ip>:/workspace/data/
```

## Step 3: Create Configuration File

Generate a config file based on your data type:

### For JSON Data (SFT):
```bash
python -m finetune_project init \
    --output /workspace/configs/my_config.yaml \
    --template json_sft
```

### For PDF Data (Causal LM):
```bash
python -m finetune_project init \
    --output /workspace/configs/my_config.yaml \
    --template pdf_causal_lm
```

### For PDF Data (QA Generation):
```bash
python -m finetune_project init \
    --output /workspace/configs/my_config.yaml \
    --template qa_generation
```

## Step 4: Edit Configuration

Open and customize the config file:

```bash
nano /workspace/configs/my_config.yaml
```

### Key Settings to Configure:

```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # Change to your preferred model
  lora_rank: 16                            # Higher = more parameters to train
  lora_alpha: 32                           # Typically 2x rank
  use_qlora: true                          # Use 4-bit quantization

data:
  input_path: "/workspace/data"            # Your data location
  output_path: "/workspace/processed_data"
  input_type: "pdf"                        # "json", "pdf", "docx", "txt"

  # PDF-specific settings
  pdf:
    extraction_backend: "pymupdf"          # "pymupdf", "pdfplumber"
    chunk_size: 1024                       # Tokens per chunk
    chunk_overlap: 128                     # Overlap between chunks
    splitter_type: "recursive"             # "recursive", "character", "token"

  # LangChain settings
  langchain:
    enabled: true
    qa_generation_enabled: false           # Set true for QA generation
    qa_llm_provider: "anthropic"           # "anthropic" or "openai"
    qa_pairs_per_chunk: 3                  # Q&A pairs per document chunk

  # Augmentation settings
  augmentation:
    enabled: false                         # Set true to enable
    instruction_variations: true
    num_instruction_variations: 2

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 2048
  output_dir: "/workspace/output"
```

## Step 5: Validate Configuration

Check that your config is valid:

```bash
python -m finetune_project validate --config /workspace/configs/my_config.yaml
```

## Step 6: Process Data

### Basic Processing (No QA Generation):

```bash
python -m finetune_project prepare-data \
    --config /workspace/configs/my_config.yaml
```

### With QA Generation (Requires ANTHROPIC_API_KEY):

```bash
# Make sure API key is set
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

python -m finetune_project prepare-data \
    --config /workspace/configs/my_config.yaml \
    --enable-qa
```

### With Data Augmentation:

```bash
python -m finetune_project prepare-data \
    --config /workspace/configs/my_config.yaml \
    --augment
```

### With Both QA and Augmentation:

```bash
python -m finetune_project prepare-data \
    --config /workspace/configs/my_config.yaml \
    --enable-qa \
    --augment
```

**Processing time**:
- JSON: ~1-5 minutes for 10K samples
- PDF without QA: ~5-15 minutes for 100 pages
- PDF with QA: ~20-60 minutes for 100 pages (depends on API speed)

## Step 7: Verify Processed Data

Check the processed data:

```bash
# View first few lines
head -n 5 /workspace/processed_data/train.jsonl

# Count samples
wc -l /workspace/processed_data/train.jsonl

# View data statistics
python -c "
import json
with open('/workspace/processed_data/train.jsonl') as f:
    samples = [json.loads(line) for line in f]
    print(f'Total samples: {len(samples)}')
    print(f'First sample:')
    print(json.dumps(samples[0], indent=2))
"
```

## Data Processing Options Explained

### 1. Direct Processing (No QA)
- **Use case**: You already have Q&A pairs or want causal language modeling
- **Speed**: Fast
- **Cost**: Free
- **Output**: Formatted training data

### 2. QA Generation with Claude
- **Use case**: Convert documents into Q&A training data
- **Speed**: Slower (API calls)
- **Cost**: ~$0.25 per 1M tokens (Claude Haiku)
- **Output**: Generated Q&A pairs from documents

### 3. Data Augmentation
- **Use case**: Increase dataset size with variations
- **Speed**: Fast (deterministic, no API)
- **Cost**: Free
- **Output**: Original data + variations

## Troubleshooting

### Issue: PDF parsing errors
```bash
# Try different extraction backend in config.yaml
pdf:
  extraction_backend: "pdfplumber"  # Instead of "pymupdf"
```

### Issue: QA generation fails
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test API connection
python -c "
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model='claude-3-haiku-20240307')
print(llm.invoke('test'))
"
```

### Issue: Out of memory during processing
```bash
# Reduce chunk size in config
pdf:
  chunk_size: 512  # Smaller chunks
```

### Issue: No data processed
```bash
# Check input path
ls -la /workspace/data/

# Check file extensions match input_type
```

## Example Workflows

### Workflow 1: Simple JSON Fine-tuning
```bash
# 1. Copy JSON data
cp my_data.jsonl /workspace/data/

# 2. Create config
python -m finetune_project init --output /workspace/configs/config.yaml --template json_sft

# 3. Process
python -m finetune_project prepare-data --config /workspace/configs/config.yaml
```

### Workflow 2: PDF to QA Dataset
```bash
# 1. Upload PDFs
scp documents/*.pdf root@pod:/workspace/data/

# 2. Create config
python -m finetune_project init --output /workspace/configs/config.yaml --template qa_generation

# 3. Process with QA generation
export ANTHROPIC_API_KEY=sk-ant-xxx
python -m finetune_project prepare-data --config /workspace/configs/config.yaml --enable-qa
```

### Workflow 3: Small Dataset with Augmentation
```bash
# 1. Prepare small JSON dataset
echo '{"instruction": "Explain AI", "input": "", "output": "AI is..."}' > /workspace/data/data.jsonl

# 2. Create config with augmentation enabled
python -m finetune_project init --output /workspace/configs/config.yaml

# 3. Edit config to enable augmentation
nano /workspace/configs/config.yaml
# Set augmentation.enabled: true

# 4. Process with augmentation
python -m finetune_project prepare-data --config /workspace/configs/config.yaml --augment
```

## Next Steps

✅ Data preparation complete! Proceed to [PHASE3_TRAINING.md](PHASE3_TRAINING.md)

## Quick Reference

```bash
# Common commands
python -m finetune_project init --output config.yaml                    # Create config
python -m finetune_project validate --config config.yaml                # Validate config
python -m finetune_project prepare-data --config config.yaml            # Process data
python -m finetune_project prepare-data --config config.yaml --enable-qa # With QA
python -m finetune_project prepare-data --config config.yaml --augment  # With augmentation
```
