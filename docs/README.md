# Documentation Index

Complete step-by-step guides for fine-tuning and deploying LLMs.

## Quick Links

- 🚀 **[Google Colab Phase Guide](PHASE_COLAB.md)** - Complete step-by-step Colab walkthrough
- 📓 **[Colab Notebook](../notebooks/llm_finetune_colab.ipynb)** - Ready-to-run notebook
- 📖 **[Colab Reference Guide](COLAB_GUIDE.md)** - Comprehensive Colab documentation
- 🔧 **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common errors
- 📂 **[Using Existing JSONL Files](USING_EXISTING_JSONL.md)** - Skip data prep with existing data
- 🖥️ **RunPod/Local Setup** - Follow phase-by-phase guides below

## Overview

This documentation is organized into phases covering different platforms and stages of the fine-tuning pipeline.

## Google Colab (Beginner-Friendly)

### [Google Colab: Complete Guide](PHASE_COLAB.md)
⏱️ Time: 45-90 minutes (including training)
💰 Cost: FREE

**All-in-one Colab guide** for fine-tuning in your browser.

**You'll learn**:
- Opening and configuring Colab notebook
- Enabling free GPU (T4)
- Uploading and preparing data
- Training with progress monitoring
- Testing and downloading your model
- Troubleshooting common issues

**Outputs**:
- ✅ Fine-tuned 7B model
- ✅ Model saved to Google Drive
- ✅ Ready to deploy or use

---

## RunPod/Local Setup (Phase-by-Phase)

### [Phase 1: Environment Setup](PHASE1_SETUP.md)
⏱️ Time: 15-20 minutes

Set up your RunPod instance or local machine with all required dependencies.

**You'll learn**:
- Creating a RunPod GPU pod
- Installing dependencies (PyTorch, Transformers, LangChain)
- Configuring environment variables
- Verifying GPU and library installations

**Outputs**:
- ✅ Working development environment
- ✅ GPU access confirmed
- ✅ All libraries installed

---

### [Phase 2: Data Preparation](PHASE2_DATA_PREPARATION.md)
⏱️ Time: 10 minutes - 2 hours (depending on QA generation)

Prepare your training data using LangChain document processing.

**You'll learn**:
- Formatting JSON/JSONL data
- Processing PDF/DOCX documents
- Generating Q&A pairs with Claude
- Data augmentation techniques
- Creating configuration files

**Outputs**:
- ✅ Processed training data (JSONL format)
- ✅ Configuration file (YAML)
- ✅ Optional: Generated Q&A pairs

---

### [Phase 3: Training](PHASE3_TRAINING.md)
⏱️ Time: 30 minutes - 2 days (depending on model size and data)

Train your model using PEFT/LoRA fine-tuning.

**You'll learn**:
- Configuring training hyperparameters
- GPU memory optimization (QLoRA, gradient checkpointing)
- Running training in tmux
- Monitoring training with TensorBoard/W&B
- Troubleshooting common issues

**Outputs**:
- ✅ Trained LoRA adapter weights
- ✅ Training logs and metrics
- ✅ Model checkpoints

---

### [Phase 4: Evaluation & Export](PHASE4_EVALUATION.md)
⏱️ Time: 30 minutes - 2 hours

Evaluate your fine-tuned model and export it for deployment.

**You'll learn**:
- Testing model with sample prompts
- Comparing base vs fine-tuned performance
- Merging LoRA adapter with base model
- Exporting to different formats (GGUF, ONNX)
- Benchmarking inference speed

**Outputs**:
- ✅ Merged model (full weights)
- ✅ Exported model (GGUF/ONNX)
- ✅ Performance benchmarks
- ✅ Evaluation results

---

### [Phase 5: Deployment](PHASE5_DEPLOYMENT.md)
⏱️ Time: 1-4 hours (depending on deployment method)

Deploy your model for production use.

**You'll learn**:
- Local deployment with llama.cpp
- RunPod Serverless deployment
- HuggingFace Inference Endpoints
- vLLM for high-throughput serving
- Building custom FastAPI servers
- Production best practices

**Outputs**:
- ✅ Deployed model (API endpoint)
- ✅ Monitoring and logging configured
- ✅ Documentation for users

---

## Quick Start Paths

### Path A: Simple JSON Fine-tuning (Fastest)
1. [Phase 1: Setup](PHASE1_SETUP.md) → 15 min
2. [Phase 2: Data Prep](PHASE2_DATA_PREPARATION.md) (JSON) → 5 min
3. [Phase 3: Training](PHASE3_TRAINING.md) (7B model) → 1-3 hours
4. [Phase 4: Evaluation](PHASE4_EVALUATION.md) → 30 min
5. [Phase 5: Deployment](PHASE5_DEPLOYMENT.md) (Local) → 30 min

**Total**: ~3-5 hours

---

### Path B: PDF to QA Dataset
1. [Phase 1: Setup](PHASE1_SETUP.md) → 15 min
2. [Phase 2: Data Prep](PHASE2_DATA_PREPARATION.md) (PDF + QA) → 1-2 hours
3. [Phase 3: Training](PHASE3_TRAINING.md) → 2-5 hours
4. [Phase 4: Evaluation](PHASE4_EVALUATION.md) → 1 hour
5. [Phase 5: Deployment](PHASE5_DEPLOYMENT.md) (RunPod) → 2 hours

**Total**: ~7-11 hours

---

### Path C: Production Deployment
1. [Phase 1: Setup](PHASE1_SETUP.md) → 15 min
2. [Phase 2: Data Prep](PHASE2_DATA_PREPARATION.md) → Variable
3. [Phase 3: Training](PHASE3_TRAINING.md) → Variable
4. [Phase 4: Evaluation](PHASE4_EVALUATION.md) → 2 hours
5. [Phase 5: Deployment](PHASE5_DEPLOYMENT.md) (vLLM/FastAPI) → 4 hours

**Total**: Variable + ~7 hours

---

## Additional Resources

### Troubleshooting & Help
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive error solutions
  - GitHub authentication (Colab)
  - CUDA out of memory errors
  - Training and data issues
  - Installation problems
  - Platform-specific fixes

### Architecture & Diagrams
- [ARCHITECTURE.md](../ARCHITECTURE.md) - ASCII diagrams and technical architecture
- [DIAGRAMS.md](../DIAGRAMS.md) - Mermaid flowcharts (renders on GitHub)

### Data Guides
- [USING_EXISTING_JSONL.md](USING_EXISTING_JSONL.md) - Use pre-existing JSONL files
- Phase 2 guide covers creating JSONL from scratch

### Project Files
- [README.md](../README.md) - Main project README
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Community guidelines

### Example Configs
- `configs/existing_jsonl.yaml` - Use existing JSONL files
- `configs/existing_jsonl_colab.yaml` - Colab with existing JSONL
- `configs/existing_jsonl_skip_prep.yaml` - Skip all data prep
- `configs/json_sft.yaml` - JSON supervised fine-tuning
- `configs/pdf_causal_lm.yaml` - PDF continued pretraining
- `configs/qa_generation.yaml` - PDF to Q&A generation
- `configs/vedic_astrology.yaml` - Domain-specific example

---

## Support & Community

### Getting Help

1. **Check the docs first** - Most questions are answered in these guides
2. **GitHub Issues** - Report bugs or request features
3. **Discussions** - Ask questions, share results

### Reporting Issues

When reporting issues, include:
- Phase you're on
- Full error message
- Configuration file (sanitized)
- GPU type and VRAM
- Steps to reproduce

---

## Tips for Success

### Before You Start
- [ ] Estimate total time needed
- [ ] Check GPU availability and cost
- [ ] Prepare your dataset
- [ ] Set realistic expectations

### During Training
- [ ] Use tmux to prevent interruptions
- [ ] Monitor GPU utilization
- [ ] Save checkpoints frequently
- [ ] Watch for OOM errors early

### After Training
- [ ] Test thoroughly before deploying
- [ ] Compare with base model
- [ ] Document your findings
- [ ] Plan for iteration

---

## What to Expect

### Timeline (7B Model, 10K Samples)
- **Day 1**: Setup, data prep, start training (4-6 hours active)
- **Day 2**: Training completes, evaluation (2-3 hours active)
- **Day 3**: Deployment and testing (3-4 hours active)

### Costs (RunPod, 7B Model)
- **Setup**: Free
- **Training**: $5-20 (depending on duration)
- **Deployment**: $0.44/hour (serverless cheaper)

### Expected Results
- Training loss: Should drop from ~3.0 to <1.0
- Evaluation: Model should outperform base on your domain
- Inference: 20-40 tokens/sec on consumer GPU

---

## Next Steps

Start with [Phase 1: Environment Setup](PHASE1_SETUP.md) →

Good luck with your fine-tuning journey! 🚀
