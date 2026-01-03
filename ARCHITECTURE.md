# Architecture and Flow Diagrams

This document provides visual representations of the LLM Fine-Tuning Pipeline architecture and workflows.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM Fine-Tuning Pipeline                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │   JSON   │   │   PDF    │   │   DOCX   │   │   TXT    │            │
│  │  Files   │   │  Files   │   │  Files   │   │  Files   │            │
│  └─────┬────┘   └─────┬────┘   └─────┬────┘   └─────┬────┘            │
│        └──────────────┴──────────────┴──────────────┘                   │
│                            │                                             │
└────────────────────────────┼─────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA PROCESSING LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │              LangChain Document Pipeline                       │      │
│  ├───────────────────────────────────────────────────────────────┤      │
│  │                                                                 │      │
│  │  ┌─────────────────┐    ┌─────────────────┐                  │      │
│  │  │ Document Loaders│    │ Text Splitters  │                  │      │
│  │  ├─────────────────┤    ├─────────────────┤                  │      │
│  │  │ • PDFLoader     │───▶│ • Recursive     │                  │      │
│  │  │ • DocxLoader    │    │ • Character     │                  │      │
│  │  │ • TextLoader    │    │ • Token-based   │                  │      │
│  │  └─────────────────┘    └────────┬────────┘                  │      │
│  │                                   │                            │      │
│  └───────────────────────────────────┼────────────────────────────┘      │
│                                      │                                   │
│                    ┌─────────────────┴─────────────────┐                │
│                    │                                     │                │
│                    ▼                                     ▼                │
│  ┌──────────────────────────────┐      ┌──────────────────────────────┐ │
│  │   Direct Processing          │      │   QA Generation (Optional)   │ │
│  ├──────────────────────────────┤      ├──────────────────────────────┤ │
│  │ • Format as training data    │      │ • Claude Haiku API           │ │
│  │ • Apply prompt templates     │      │ • Generate Q&A pairs         │ │
│  │ • Chunk for causal LM        │      │ • 3 pairs per chunk          │ │
│  └──────────────┬───────────────┘      └──────────────┬───────────────┘ │
│                 │                                       │                 │
│                 └───────────────┬───────────────────────┘                 │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │              Data Augmentation Engine                        │         │
│  ├─────────────────────────────────────────────────────────────┤         │
│  │ • Instruction variations (template-based)                   │         │
│  │ • Paraphrase templates (rule-based)                         │         │
│  │ • Whitespace normalization                                  │         │
│  │ • Case variations                                            │         │
│  └─────────────────────────────┬───────────────────────────────┘         │
│                                 │                                         │
└─────────────────────────────────┼─────────────────────────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Formatted Data  │
                        │   (JSONL/TXT)    │
                        └────────┬─────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │                  Model Configuration                           │      │
│  ├───────────────────────────────────────────────────────────────┤      │
│  │  Base Models:                                                  │      │
│  │  • Llama 3.x (7B, 8B, 70B)                                    │      │
│  │  • Qwen 2.5 (7B, 14B, 32B)                                    │      │
│  │  • Mistral (7B, 8x7B)                                         │      │
│  └───────────────────────────┬───────────────────────────────────┘      │
│                               │                                          │
│                               ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │                  PEFT/LoRA Configuration                       │      │
│  ├───────────────────────────────────────────────────────────────┤      │
│  │ • LoRA rank: 8-64                                             │      │
│  │ • LoRA alpha: 16-128                                          │      │
│  │ • Target modules: q_proj, v_proj, k_proj, o_proj             │      │
│  │ • Dropout: 0.05-0.1                                           │      │
│  │ • QLoRA: 4-bit quantization (optional)                        │      │
│  └───────────────────────────┬───────────────────────────────────┘      │
│                               │                                          │
│                               ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │                    TRL SFTTrainer                              │      │
│  ├───────────────────────────────────────────────────────────────┤      │
│  │ • Supervised Fine-Tuning                                      │      │
│  │ • Sequence packing support                                    │      │
│  │ • Chat template formatting                                    │      │
│  │ • Gradient checkpointing                                      │      │
│  │ • Mixed precision (bf16/fp16)                                 │      │
│  └───────────────────────────┬───────────────────────────────────┘      │
│                               │                                          │
└───────────────────────────────┼──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  LoRA        │   │   Merged     │   │   Exported   │                │
│  │  Adapter     │──▶│   Model      │──▶│   Model      │                │
│  │  Weights     │   │              │   │  (GGUF/etc)  │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Processing Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Data Preparation Workflow                            │
└─────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌────────────────────────────┐
│  Load Configuration File   │
│  (config.yaml)             │
└──────────┬─────────────────┘
           │
           ▼
      ┌─────────┐
      │ Input   │
      │ Type?   │
      └────┬────┘
           │
  ─────────┴─────────────────────────────────
  │                │                          │
  ▼                ▼                          ▼
┌──────┐      ┌──────┐                  ┌──────────┐
│ JSON │      │ PDF  │                  │ DOCX/TXT │
└──┬───┘      └───┬──┘                  └─────┬────┘
   │              │                            │
   │              ▼                            │
   │      ┌──────────────────┐                │
   │      │ LangChain Loader │                │
   │      │ • PDFLoader      │◀───────────────┘
   │      │ • DocxLoader     │
   │      │ • TextLoader     │
   │      └────────┬─────────┘
   │               │
   │               ▼
   │      ┌──────────────────┐
   │      │  Text Splitting  │
   │      │ • Recursive      │
   │      │ • Character      │
   │      │ • Token-based    │
   │      └────────┬─────────┘
   │               │
   │               ▼
   │         ┌───────────┐
   │         │ QA Gen?   │
   │         └─────┬─────┘
   │          No   │   Yes
   │      ┌────────┴────────┐
   │      │                 │
   │      ▼                 ▼
   │   ┌────────┐   ┌──────────────────┐
   │   │ Direct │   │ Generate Q&A     │
   │   │ Format │   │ via Claude Haiku │
   │   └───┬────┘   └────────┬─────────┘
   │       │                 │
   └───────┴─────────┬───────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Apply Prompt        │
          │  Templates           │
          │ • Alpaca             │
          │ • ChatML             │
          │ • Custom             │
          └──────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Augment?    │
              └──────┬──────┘
                No   │   Yes
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
        ┌────────┐   ┌──────────────────┐
        │  Save  │   │  Data Augment    │
        │        │◀──│ • Variations     │
        │        │   │ • Paraphrasing   │
        └────┬───┘   └──────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Save to JSONL/TXT  │
    │ Ready for Training │
    └────────────────────┘
             │
             ▼
           END
```

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline Flow                            │
└─────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌────────────────────────────┐
│ Load Training Config       │
│ • Model name               │
│ • LoRA parameters          │
│ • Training hyperparams     │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Initialize Base Model      │
│ • Download from HF Hub     │
│ • Apply quantization       │
│   (QLoRA if enabled)       │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Configure LoRA Adapter     │
│ • Set rank & alpha         │
│ • Select target modules    │
│ • Set dropout              │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Load Prepared Dataset      │
│ • Read JSONL/TXT           │
│ • Apply tokenization       │
│ • Pack sequences           │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Initialize SFTTrainer      │
│ • Set batch size           │
│ • Configure optimizer      │
│ • Enable grad checkpoint   │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│   Training Loop            │
│ ┌────────────────────────┐ │
│ │ For each epoch:        │ │
│ │  • Forward pass        │ │
│ │  • Compute loss        │ │
│ │  • Backward pass       │ │
│ │  • Update weights      │ │
│ │  • Log metrics         │ │
│ └────────────────────────┘ │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Save LoRA Adapter          │
│ • adapter_config.json      │
│ • adapter_model.safetensors│
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Optional: Merge Adapter    │
│ • Combine with base model  │
│ • Save full model          │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Optional: Export           │
│ • Convert to GGUF          │
│ • Quantize for inference   │
└──────────┬─────────────────┘
           │
           ▼
          END
```

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Component Architecture                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  cli.py (Typer CLI)                                          │       │
│  ├──────────────────────────────────────────────────────────────┤       │
│  │  Commands:                                                    │       │
│  │  • init          - Initialize config                         │       │
│  │  • validate      - Validate config                           │       │
│  │  • prepare-data  - Process input data                        │       │
│  │  • train         - Start training                            │       │
│  │  • evaluate      - Evaluate model                            │       │
│  │  • generate-qa   - Generate QA pairs                         │       │
│  │  • merge-adapter - Merge LoRA with base                      │       │
│  │  • export        - Export model                              │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                           │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Configuration Layer                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  config.py                                                    │       │
│  ├──────────────────────────────────────────────────────────────┤       │
│  │  Classes:                                                     │       │
│  │  • ModelConfig       - Model & LoRA settings                 │       │
│  │  • DataConfig        - Data processing settings              │       │
│  │  • TrainingConfig    - Training hyperparameters              │       │
│  │  • LangChainConfig   - LangChain & QA settings              │       │
│  │  • AugmentConfig     - Augmentation settings                 │       │
│  │  • FineTuneConfig    - Main config container                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                           │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Processing Layer                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐       │
│  │ langchain_pipeline.py     │  │  prompt_templates.py          │       │
│  ├───────────────────────────┤  ├───────────────────────────────┤       │
│  │ • load_documents()        │  │ • PromptTemplateManager       │       │
│  │ • split_text()            │  │ • format_alpaca()             │       │
│  │ • generate_qa_pairs()     │  │ • format_chatml()             │       │
│  │ • process_pipeline()      │  │ • format_custom()             │       │
│  └───────────────────────────┘  │ • get_instruction_variations()│       │
│                                  └───────────────────────────────┘       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │  augmentation.py                                              │       │
│  ├───────────────────────────────────────────────────────────────┤       │
│  │  Classes:                                                     │       │
│  │  • DataAugmenter                                             │       │
│  │                                                               │       │
│  │  Methods:                                                     │       │
│  │  • augment_instruction() - Vary instructions                 │       │
│  │  • paraphrase()          - Template-based paraphrase         │       │
│  │  • normalize()           - Whitespace/case normalization     │       │
│  │  • augment_dataset()     - Process full dataset              │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Training Layer                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │  trainer.py                                                    │       │
│  ├───────────────────────────────────────────────────────────────┤       │
│  │  Functions:                                                    │       │
│  │  • load_model()          - Load base model with quant         │       │
│  │  • configure_lora()      - Setup PEFT config                  │       │
│  │  • load_dataset()        - Load & tokenize data               │       │
│  │  • train()               - Main training function             │       │
│  │  • evaluate()            - Model evaluation                   │       │
│  │  • merge_adapter()       - Merge LoRA weights                 │       │
│  │  • export_model()        - Export to GGUF/etc                 │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        External Dependencies                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Transformers│  │  LangChain  │  │    PEFT     │  │     TRL     │   │
│  │   (HF)      │  │             │  │    (LoRA)   │  │ (SFTTrainer)│   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Anthropic │  │    Typer    │  │   PyYAML    │  │   Datasets  │   │
│  │   (Claude)  │  │    (CLI)    │  │   (Config)  │  │    (HF)     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## RunPod Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RunPod Infrastructure                             │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  RunPod GPU Pod                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Container: RunPod PyTorch 2.x                        │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │                                                        │      │
│  │  GPU: RTX 4090 / A40 / A100                          │      │
│  │  VRAM: 24GB / 46GB / 40GB                            │      │
│  │                                                        │      │
│  │  ┌────────────────────────────────────────────┐      │      │
│  │  │  /workspace (Persistent Volume)            │      │
│  │  ├────────────────────────────────────────────┤      │      │
│  │  │                                             │      │      │
│  │  │  ├─ llm-finetune-langchain/               │      │      │
│  │  │  │  ├─ src/                                │      │      │
│  │  │  │  ├─ configs/                            │      │      │
│  │  │  │  ├─ scripts/                            │      │      │
│  │  │  │  └─ data/                               │      │      │
│  │  │  │                                          │      │      │
│  │  │  ├─ data/           (Input files)          │      │      │
│  │  │  ├─ output/         (LoRA adapters)        │      │      │
│  │  │  ├─ merged-model/   (Merged weights)       │      │      │
│  │  │  └─ hf_cache/       (Model cache)          │      │      │
│  │  │                                             │      │      │
│  │  └────────────────────────────────────────────┘      │      │
│  │                                                        │      │
│  │  Environment Variables:                               │      │
│  │  • HF_TOKEN                                           │      │
│  │  • ANTHROPIC_API_KEY (optional)                      │      │
│  │  • WANDB_API_KEY (optional)                          │      │
│  │                                                        │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
                             │
                             │ SSH/HTTP Connect
                             │
                             ▼
                    ┌─────────────────┐
                    │  Developer      │
                    │  • SSH access   │
                    │  • Jupyter      │
                    │  • Web terminal │
                    └─────────────────┘
```

## Data Flow Summary

```
Raw Data → LangChain → Templates → Augmentation → Training → Fine-tuned Model
  (PDF/     (Process/   (Format)   (Variations)   (LoRA)     (Adapter/
  JSON/      Split/                                           Merged)
  DOCX)      QA Gen)
```

## Key Design Patterns

1. **Pipeline Pattern**: Sequential data processing stages
2. **Strategy Pattern**: Pluggable document loaders and text splitters
3. **Template Method**: Prompt template management
4. **Facade Pattern**: CLI simplifies complex operations
5. **Configuration Pattern**: YAML-based declarative configuration
