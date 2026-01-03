# Phase 3: Model Training

This guide covers the training phase of the fine-tuning pipeline.

## Prerequisites

✅ Completed [PHASE1_SETUP.md](PHASE1_SETUP.md)
✅ Completed [PHASE2_DATA_PREPARATION.md](PHASE2_DATA_PREPARATION.md)

## Step 1: Review Training Configuration

Open your config file and verify training settings:

```bash
nano /workspace/configs/my_config.yaml
```

Key training parameters:

```yaml
training:
  # Training duration
  num_epochs: 3                        # Number of complete passes through data
  max_steps: -1                        # Override epochs if > 0

  # Batch sizes (adjust based on VRAM)
  per_device_train_batch_size: 2       # Batch size per GPU
  gradient_accumulation_steps: 8       # Effective batch = 2 * 8 = 16

  # Learning rates
  learning_rate: 2.0e-4                # LoRA learning rate
  warmup_ratio: 0.03                   # Warmup steps (3% of total)
  lr_scheduler_type: "cosine"          # Learning rate schedule

  # Sequence length
  max_seq_length: 2048                 # Maximum token length

  # Memory optimization
  gradient_checkpointing: true         # Saves VRAM, slightly slower
  fp16: false                          # Use fp16 precision
  bf16: true                           # Use bf16 (better on Ampere+ GPUs)

  # Saving
  output_dir: "/workspace/output"
  save_strategy: "steps"               # Save checkpoints by steps
  save_steps: 100                      # Save every 100 steps
  save_total_limit: 3                  # Keep only 3 checkpoints

  # Logging
  logging_steps: 10                    # Log every 10 steps
  report_to: ["tensorboard"]           # Options: tensorboard, wandb

  # Evaluation (optional)
  evaluation_strategy: "no"            # "steps" or "epoch" to enable
  eval_steps: 100                      # Evaluate every N steps
```

## Step 2: Adjust Settings for Your GPU

### For RTX 4090 (24GB VRAM):

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  gradient_checkpointing: true
  bf16: true

model:
  use_qlora: true                      # 4-bit quantization
  lora_rank: 16
```

### For A40 (46GB VRAM):

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  gradient_checkpointing: true
  bf16: true

model:
  use_qlora: false                     # Can use full precision
  lora_rank: 32
```

### For A100 (40GB/80GB VRAM):

```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  max_seq_length: 4096
  gradient_checkpointing: false
  bf16: true

model:
  use_qlora: false
  lora_rank: 64
```

## Step 3: Start Training in tmux (Recommended)

Using tmux ensures training continues if you disconnect:

```bash
# Create new tmux session
tmux new -s training

# Navigate to project
cd /workspace/llm-finetune

# Start training
python -m finetune_project train --config /workspace/configs/my_config.yaml

# Detach from tmux: Press Ctrl+B, then D
```

Re-attach to session:
```bash
tmux attach -t training
```

## Step 4: Monitor Training

### Option A: Watch Logs (in tmux)

Training logs show:
- Current step / total steps
- Loss values
- Learning rate
- GPU memory usage
- Estimated time remaining

### Option B: TensorBoard

In a new terminal/tmux window:

```bash
# Start TensorBoard
tensorboard --logdir /workspace/output --host 0.0.0.0 --port 6006

# On RunPod: Access via pod's HTTP endpoint
# URL: http://<pod-id>-6006.proxy.runpod.net
```

### Option C: Weights & Biases (W&B)

If you set up W&B in Phase 1:

1. Update config:
```yaml
training:
  report_to: ["wandb"]
```

2. Login to W&B:
```bash
wandb login $WANDB_API_KEY
```

3. Training metrics will appear at: https://wandb.ai

## Step 5: Training Commands Reference

### Basic Training:
```bash
python -m finetune_project train --config /workspace/configs/my_config.yaml
```

### Resume from Checkpoint:
```bash
python -m finetune_project train \
    --config /workspace/configs/my_config.yaml \
    --resume-from-checkpoint /workspace/output/checkpoint-100
```

### Override Config Settings:
```bash
python -m finetune_project train \
    --config /workspace/configs/my_config.yaml \
    --learning-rate 3e-4 \
    --num-epochs 5
```

## Step 6: Training Time Estimates

Approximate training times (varies by data size and hardware):

### 7B Model on RTX 4090:
- 1K samples, 3 epochs: ~30-45 minutes
- 10K samples, 3 epochs: ~3-5 hours
- 100K samples, 3 epochs: ~24-30 hours

### 14B Model on A40:
- 1K samples, 3 epochs: ~1-1.5 hours
- 10K samples, 3 epochs: ~8-12 hours
- 100K samples, 3 epochs: ~3-4 days

### Factors affecting speed:
- Sequence length (longer = slower)
- Batch size (larger = faster but more VRAM)
- LoRA rank (higher = more parameters = slower)
- Flash Attention (2-3x faster if available)

## Step 7: During Training - What to Watch

### Good Signs ✅:
- Loss steadily decreasing
- GPU utilization >90%
- No out-of-memory errors
- Checkpoints saving successfully

### Warning Signs ⚠️:
- Loss increasing or erratic → Lower learning rate
- Loss stuck/not changing → Increase learning rate
- GPU utilization <50% → Increase batch size
- Frequent OOM errors → See troubleshooting below

### Training Curves:

```
Loss should look like:
3.5 |⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
3.0 |⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
2.5 |⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
2.0 |⠀⠈⢢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
1.5 |⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
1.0 |⠀⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀
0.5 |⠀⠀⠀⠀⠀⠀⠉⠒⠤⣀⡀⠀⠀⠀
0.0 |⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠒⠒
    └─────────────────────
    0    Steps    1000
```

## Step 8: When Training Completes

Training outputs are saved to `/workspace/output/`:

```
/workspace/output/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # Trained LoRA weights
├── trainer_state.json           # Training state
├── training_args.bin            # Training arguments
├── tokenizer_config.json        # Tokenizer config
├── special_tokens_map.json      # Special tokens
└── checkpoint-XXX/              # Intermediate checkpoints
```

## Troubleshooting

### Out of Memory (OOM)

Try these in order:

1. **Reduce batch size**:
```yaml
training:
  per_device_train_batch_size: 1  # Reduce from 2
```

2. **Enable gradient checkpointing**:
```yaml
training:
  gradient_checkpointing: true
```

3. **Reduce sequence length**:
```yaml
training:
  max_seq_length: 1024  # Reduce from 2048
```

4. **Enable QLoRA**:
```yaml
model:
  use_qlora: true  # 4-bit quantization
```

5. **Reduce LoRA rank**:
```yaml
model:
  lora_rank: 8  # Reduce from 16
```

### Training Too Slow

1. **Increase batch size** (if VRAM available):
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

2. **Check Flash Attention**:
```bash
python -c "import flash_attn; print('Flash Attention available')"
```

3. **Disable unnecessary logging**:
```yaml
training:
  logging_steps: 50  # Log less frequently
```

### Loss Not Decreasing

1. **Increase learning rate**:
```yaml
training:
  learning_rate: 5.0e-4  # Increase from 2e-4
```

2. **Check data quality**:
```bash
# Inspect processed data
head -n 10 /workspace/processed_data/train.jsonl
```

3. **Increase LoRA rank**:
```yaml
model:
  lora_rank: 32  # Increase from 16
```

### CUDA Errors

```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Restart from checkpoint
python -m finetune_project train \
    --config config.yaml \
    --resume-from-checkpoint /workspace/output/checkpoint-XXX
```

## Next Steps

✅ Training complete! Proceed to [PHASE4_EVALUATION.md](PHASE4_EVALUATION.md)

## Quick Reference

```bash
# Start training
python -m finetune_project train --config config.yaml

# Resume training
python -m finetune_project train --config config.yaml --resume-from-checkpoint checkpoint-100

# Monitor GPU
watch -n 1 nvidia-smi

# View TensorBoard
tensorboard --logdir /workspace/output --host 0.0.0.0 --port 6006

# Check training progress
tail -f /workspace/output/trainer_state.json

# tmux commands
tmux new -s training        # Create session
Ctrl+B, D                   # Detach
tmux attach -t training     # Reattach
tmux kill-session -t training  # Kill session
```
