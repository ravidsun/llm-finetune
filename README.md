# LLM Fine-Tuning Pipeline with LangChain

A production-ready pipeline for fine-tuning open-source LLMs (Llama, Qwen, Mistral) on RunPod with **LangChain integration** for document processing, prompt templating, and optional LLM-based QA generation.

> **⚡ Quick Start**: Use our [automated RunPod script](#-runpod-automated---recommended) for one-command setup!
> **📊 Visual Guides**: Check out [ARCHITECTURE.md](ARCHITECTURE.md) for ASCII diagrams and [DIAGRAMS.md](DIAGRAMS.md) for interactive Mermaid flowcharts.

## Features

### Core Features
- **Automated Setup**: One-command installation and training on RunPod
- **Multiple Models**: Support for Llama, Qwen, Mistral, and other Hugging Face models
- **PEFT/LoRA**: Parameter-efficient fine-tuning with configurable LoRA adapters
- **QLoRA**: 4-bit quantization for training larger models on limited VRAM
- **GPU Auto-Detection**: Automatically optimizes settings based on available VRAM
- **Multiple JSONL Support**: Easily use existing training data files

### LangChain Integration
- **Document Loading**: PDF, DOCX, TXT via LangChain loaders
- **Intelligent Chunking**: RecursiveCharacterTextSplitter, TokenTextSplitter
- **Prompt Templates**: Built-in Alpaca, ChatML, and custom templates
- **QA Generation**: Optional Claude Haiku-powered Q&A pair generation from documents

### Data Augmentation
- **Deterministic Augmentation**: No API calls required
- **Instruction Variations**: Template-based instruction rephrasing
- **Paraphrase Templates**: Rule-based text variations
- **Fully Configurable**: Enable/disable per augmentation type

## Quick Start

### 🚀 RunPod (Automated - Recommended)

**Complete automated setup + training in one command:**

```bash
# On RunPod, run this single command for everything:
curl -fsSL https://raw.githubusercontent.com/ravidsun/llm-finetune/master/scripts/runpod.sh | bash
```

This will automatically:
- ✅ Install all dependencies
- ✅ Clone repository
- ✅ Set up directories
- ✅ Verify GPU access
- ✅ Start training (after you upload data)

**Or just setup first:**
```bash
curl -fsSL https://raw.githubusercontent.com/ravidsun/llm-finetune/master/scripts/runpod.sh | bash -s setup
# Then later: bash scripts/runpod.sh train
```

📖 **[Full RunPod Guide](RUNPOD_GUIDE.md)** - Step-by-step with automation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ravidsun/llm-finetune.git
cd llm-finetune

# Install core dependencies
pip install -e .

# Install with LLM providers (for QA generation)
pip install -e ".[llm]"
```

### 2. Prepare Your Data (Local/Manual Setup)

#### Option A: Use Existing JSONL Files (Easiest!)

**Already have training data?** Just use it directly!

```bash
# Place your existing JSONL file(s) - supports multiple files
cp your-data.jsonl data/input/train.jsonl
# Or multiple files:
cp *.jsonl data/input/

# Use the pre-configured setup
python -m finetune_project init --output config.yaml --template existing_jsonl
```

📖 **See**: [Phase 2: Data Preparation](docs/PHASE2_DATA_PREPARATION.md#using-existing-jsonl-files) for detailed guide

#### Option B: Create New Data

**JSON format (recommended for instruction tuning):**
```jsonl
{"instruction": "What is the capital of France?", "input": "", "output": "Paris"}
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses..."}
```

**PDF documents (for continued pretraining or QA generation):**
- Place PDFs in `data/input/`
- The pipeline extracts text and chunks using LangChain

### 3. Configure and Train (Local/Manual Setup)

```bash
# Generate config
python -m finetune_project init --output config.yaml --template existing_jsonl

# Prepare data
python -m finetune_project prepare-data --config config.yaml

# Optional: Enable QA generation from PDFs (requires ANTHROPIC_API_KEY)
python -m finetune_project prepare-data --config config.yaml --enable-qa

# Train
python -m finetune_project train --config config.yaml
```

## CLI Commands

```bash
# Initialize config
python -m finetune_project init --output config.yaml
python -m finetune_project init --output config.yaml --template qa_generation

# Validate config
python -m finetune_project validate --config config.yaml

# Prepare data
python -m finetune_project prepare-data --config config.yaml
python -m finetune_project prepare-data --config config.yaml --enable-qa
python -m finetune_project prepare-data --config config.yaml --augment

# Train
python -m finetune_project train --config config.yaml

# Evaluate
python -m finetune_project evaluate --config config.yaml

# Generate QA pairs (standalone)
python -m finetune_project generate-qa \
    --input /path/to/documents \
    --output qa_pairs.jsonl \
    --num-pairs 3 \
    --provider anthropic

# Merge adapter
python -m finetune_project merge-adapter \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter /workspace/output \
    --output /workspace/merged-model

# Export
python -m finetune_project export \
    --model /workspace/merged-model \
    --output /workspace/export
```

## LangChain Features

### Document Processing

LangChain loaders support multiple document types:

```yaml
data:
  input_type: "pdf"  # or "docx", "json", "auto"
  
  pdf:
    extraction_backend: "pymupdf"  # pymupdf, pdfplumber, unstructured
    chunk_size: 1024
    chunk_overlap: 128
    splitter_type: "recursive"  # recursive, character, token
    separators:
      - "\n\n"
      - "\n"
      - ". "
      - " "
```

### Prompt Templates

Built-in templates with LangChain prompt management:

```python
from finetune_project.prompt_templates import PromptTemplateManager

manager = PromptTemplateManager()

# Alpaca format
text = manager.format_alpaca(
    instruction="Explain machine learning",
    input_text="",
    output="Machine learning is..."
)

# Get instruction variations for augmentation
variations = manager.get_instruction_variations(
    "Explain quantum computing",
    num_variations=3
)
```

### QA Generation (Optional)

Generate Q&A pairs from documents using Claude Haiku:

```yaml
data:
  langchain:
    # QA Generation (requires ANTHROPIC_API_KEY)
    qa_generation_enabled: true
    qa_llm_provider: "anthropic"
    qa_model_name: "claude-3-haiku-20240307"
    qa_pairs_per_chunk: 3
    qa_temperature: 0.3
```

**Important**: QA generation is **OFF by default**. Enable with `--enable-qa` flag or config.

```bash
# Generate QA pairs
export ANTHROPIC_API_KEY=your_key
python -m finetune_project prepare-data --config config.yaml --enable-qa
```

### Data Augmentation

Deterministic augmentation (no API calls):

```yaml
data:
  augmentation:
    enabled: true
    instruction_variations: true
    num_instruction_variations: 2
    whitespace_normalization: true
    use_paraphrase_templates: true
    case_variations: false
```

## RunPod Setup

### 1. Create a GPU Pod

1. Go to [RunPod](https://runpod.io) and create an account
2. Add credits ($10-15 recommended)
3. Deploy a GPU Pod:
   - **GPU**: RTX 4090 (24GB) or A40 (46GB)
   - **Template**: RunPod PyTorch 2.x
   - **Container Disk**: 30GB
   - **Volume Disk**: 50GB at `/workspace`

### 2. Recommended GPUs

| Model Size | GPU | VRAM | Approx. Cost |
|------------|-----|------|--------------|
| 7B (QLoRA) | RTX 4090 | 24GB | $0.44/hr |
| 14B (QLoRA) | A40 | 46GB | $0.39/hr |
| 32B (QLoRA) | A100 40GB | 40GB | $1.89/hr |

### 3. Environment Variables

```bash
# Required for gated models
export HF_TOKEN=your_huggingface_token

# Optional: QA generation
export ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Experiment tracking
export WANDB_API_KEY=your_wandb_key
```

### 4. Automated Training

**Using the unified script:**

```bash
cd /workspace/llm-finetune

# Upload your JSONL files to data/input/ first, then:
bash scripts/runpod.sh train

# The script will:
# ✅ Auto-detect your GPU and optimize settings
# ✅ Generate optimized config automatically
# ✅ Process your data
# ✅ Start training in tmux session
# ✅ Show monitoring commands
```

**Monitor training:**

```bash
# Attach to training session
tmux attach -t training

# View logs
tail -f output/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Configuration Reference

### LangChain Configuration

```yaml
data:
  langchain:
    enabled: true
    
    # QA Generation (OFF by default)
    qa_generation_enabled: false
    qa_llm_provider: "anthropic"  # or "openai"
    qa_model_name: "claude-3-haiku-20240307"
    qa_pairs_per_chunk: 3
    qa_temperature: 0.3
    qa_max_tokens: 1024
    
    # Custom templates
    prompt_template_dir: null
    default_system_message: null
```

### Augmentation Configuration

```yaml
data:
  augmentation:
    enabled: false  # OFF by default
    instruction_variations: true
    num_instruction_variations: 2
    case_variations: false
    whitespace_normalization: true
    use_paraphrase_templates: true
```

### PDF Processing Configuration

```yaml
data:
  pdf:
    extraction_backend: "pymupdf"
    chunk_size: 1024
    chunk_overlap: 128
    min_chunk_size: 100
    strip_headers_footers: true
    clean_whitespace: true
    splitter_type: "recursive"
```

## Example Configs

### 1. JSON SFT with Augmentation (`configs/json_sft.yaml`)
- Supervised fine-tuning from JSONL
- Augmentation enabled
- Best for instruction-tuning

### 2. PDF Causal LM (`configs/pdf_causal_lm.yaml`)
- Continued pretraining from PDFs
- Text-only (no QA generation)
- Packing enabled for efficiency

### 3. QA Generation (`configs/qa_generation.yaml`)
- Generate Q&A from documents
- Uses Claude Haiku
- Requires ANTHROPIC_API_KEY

### 4. Vedic Astrology (`configs/vedic_astrology.yaml`)
- Optimized for Sanskrit/multilingual
- Uses Qwen2.5-14B-Instruct
- Augmentation for small datasets

## Architecture

```
src/finetune_project/
├── __init__.py           # Package exports
├── __main__.py           # CLI entry point
├── config.py             # Configuration management
├── cli.py                # Typer CLI commands
├── langchain_pipeline.py # LangChain document processing
├── prompt_templates.py   # Prompt template management
├── augmentation.py       # Data augmentation
└── trainer.py            # Training with PEFT/LoRA
```

### Why TRL SFTTrainer?

We use TRL's SFTTrainer over base Transformers Trainer because:
- Native support for chat templates
- Built-in sequence packing
- Designed for supervised fine-tuning
- Better PEFT integration

### Why LangChain?

LangChain provides:
- Unified document loaders (PDF, DOCX, etc.)
- Intelligent text splitters
- Prompt template management
- Easy LLM integration for QA generation

## Documentation & Support

### Quick Start Guides

- 🚀 **[RunPod Automated Guide](RUNPOD_GUIDE.md)** - Comprehensive automated setup (recommended)
- 🔧 **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Solutions for common errors

### Phase-by-Phase Guides

For detailed, step-by-step instructions organized by phase:

📖 **[Complete Documentation](docs/README.md)** - Start here for guided walkthrough

1. **[Phase 1: Environment Setup](docs/PHASE1_SETUP.md)** - Install dependencies and configure environment
2. **[Phase 2: Data Preparation](docs/PHASE2_DATA_PREPARATION.md)** - Process data with LangChain (includes using existing JSONL)
3. **[Phase 3: Training](docs/PHASE3_TRAINING.md)** - Fine-tune your model with PEFT/LoRA
4. **[Phase 4: Evaluation & Export](docs/PHASE4_EVALUATION.md)** - Test and export model
5. **[Phase 5: Deployment](docs/PHASE5_DEPLOYMENT.md)** - Deploy to production

### Common Issues

**Out of Memory Errors:**
```yaml
# Reduce these values in config.yaml:
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 1024
  gradient_checkpointing: true
```

**QA Generation Issues:**
```bash
# Check API key and install dependencies
echo $ANTHROPIC_API_KEY
pip install langchain-anthropic anthropic
```

For more help, see the **[Complete Troubleshooting Guide](docs/TROUBLESHOOTING.md)**

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [LangChain](https://langchain.com) for document processing
- [Hugging Face](https://huggingface.co) for Transformers and TRL
- [RunPod](https://runpod.io) for GPU infrastructure
- [Anthropic](https://anthropic.com) for Claude API
