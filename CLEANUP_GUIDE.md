# RunPod Cleanup Guide

Complete guide for cleaning up and resetting your RunPod fine-tuning environment.

## Table of Contents

1. [Quick Cleanup Options](#quick-cleanup-options)
2. [Detailed Cleanup Scenarios](#detailed-cleanup-scenarios)
3. [Disk Space Management](#disk-space-management)
4. [Troubleshooting Cleanup](#troubleshooting-cleanup)

---

## Quick Cleanup Options

### Option 1: Clean Training Outputs Only (Recommended)

**Use when:** You want to start a fresh training run but keep everything else.

```bash
cd /workspace/llm-finetune

# Remove training outputs
rm -rf output/
rm -rf processed_data/
rm -rf config.yaml

# Kill any running training sessions
tmux kill-session -t training 2>/dev/null || true

echo "✅ Training outputs cleaned. Ready for fresh training."
```

**What gets removed:**
- ✅ Model checkpoints from previous runs
- ✅ Training logs
- ✅ Processed dataset cache
- ✅ Generated config file

**What is preserved:**
- ✅ Repository and code
- ✅ Training data files
- ✅ Downloaded models (HuggingFace cache)
- ✅ Installed dependencies

---

### Option 2: Update Code and Clean Outputs

**Use when:** You want to pull latest code updates and start fresh.

```bash
cd /workspace/llm-finetune

# Pull latest code
git pull

# Reinstall dependencies (if updated)
pip install -e .

# Clean training outputs
rm -rf output/
rm -rf processed_data/
rm -rf config.yaml

# Kill training sessions
tmux kill-session -t training 2>/dev/null || true

echo "✅ Code updated and outputs cleaned."
```

---

### Option 3: Complete Reset (Nuclear Option)

**Use when:** You want to start completely fresh or something is badly broken.

```bash
cd /workspace

# Remove entire project
rm -rf llm-finetune/

# Clear HuggingFace cache (frees significant space)
rm -rf ~/.cache/huggingface/

# Clear pip cache
pip cache purge

echo "✅ Complete cleanup done."
```

**Then reinstall:**
```bash
curl -fsSL https://raw.githubusercontent.com/ravidsun/llm-finetune/master/scripts/runpod.sh | bash
```

---

## Detailed Cleanup Scenarios

### Scenario 1: Reset Training (Keep Data)

```bash
cd /workspace/llm-finetune

# Step 1: Kill running training
tmux kill-session -t training 2>/dev/null || true

# Step 2: Remove outputs
rm -rf output/
rm -rf processed_data/
rm -rf config.yaml

# Step 3: Verify data is still there
ls -lh data/input/

# Step 4: Start fresh training
bash scripts/runpod.sh train
```

---

### Scenario 2: Change Training Data

```bash
cd /workspace/llm-finetune

# Step 1: Backup old data (optional)
mv data/input data/input.backup

# Step 2: Create fresh input directory
mkdir -p data/input

# Step 3: Upload new JSONL files
# (Use RunPod file upload or scp)

# Step 4: Clean old outputs
rm -rf output/
rm -rf processed_data/
rm -rf config.yaml

# Step 5: Start training with new data
bash scripts/runpod.sh train
```

---

### Scenario 3: Fix Broken Installation

```bash
cd /workspace/llm-finetune

# Step 1: Pull latest code
git fetch --all
git reset --hard origin/master

# Step 2: Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# Step 3: Reinstall dependencies
pip install --force-reinstall -e .

# Step 4: Clean outputs
rm -rf output/
rm -rf processed_data/
rm -rf config.yaml

echo "✅ Installation reset complete."
```

---

### Scenario 4: Free Up Disk Space

```bash
# Check disk usage
df -h /workspace

# Option A: Remove HuggingFace cache (models will re-download)
rm -rf ~/.cache/huggingface/
echo "✅ Freed ~30-50GB"

# Option B: Remove old checkpoints but keep final model
cd /workspace/llm-finetune/output
rm -rf checkpoint-*
echo "✅ Removed intermediate checkpoints"

# Option C: Clear pip cache
pip cache purge
echo "✅ Cleared pip cache"

# Option D: Clean Docker (if applicable)
docker system prune -af
echo "✅ Cleaned Docker cache"

# Check space again
df -h /workspace
```

---

## Disk Space Management

### What Takes Up Space?

| Item | Location | Typical Size | Safe to Delete? |
|------|----------|--------------|-----------------|
| HuggingFace Models | `~/.cache/huggingface/` | 30-50GB | ⚠️ Yes (will re-download) |
| Training Checkpoints | `output/checkpoint-*` | 5-15GB each | ✅ Yes (keep final only) |
| Final Model | `output/` | 15-30GB | ⚠️ Backup first |
| Training Data | `data/input/` | 1-10GB | ⚠️ Backup first |
| Processed Data | `processed_data/` | 1-5GB | ✅ Yes (regenerates) |
| Repository | `llm-finetune/` | <100MB | ⚠️ No (core code) |
| Pip Cache | `~/.cache/pip/` | 1-5GB | ✅ Yes |

### Check Disk Usage

```bash
# Overall disk usage
df -h /workspace

# Breakdown by directory
du -h --max-depth=1 /workspace/ | sort -hr

# Specific directories
du -sh ~/.cache/huggingface/
du -sh /workspace/llm-finetune/output/
du -sh /workspace/llm-finetune/data/
```

---

## Managing Training Sessions

### View Active Sessions

```bash
# List all tmux sessions
tmux list-sessions

# Check if training session exists
tmux has-session -t training 2>/dev/null && echo "Training session running" || echo "No training session"
```

### Kill Training Session

```bash
# Kill specific session
tmux kill-session -t training

# Kill all tmux sessions
tmux kill-server

# Verify
tmux list-sessions 2>/dev/null || echo "No sessions running"
```

### Attach to Running Training

```bash
# Attach to training session
tmux attach -t training

# Detach (while inside tmux)
# Press: Ctrl+B, then D
```

---

## Backup Before Cleanup

### Backup Trained Model

```bash
cd /workspace/llm-finetune

# Create backup directory
mkdir -p /workspace/backups/$(date +%Y%m%d_%H%M%S)

# Backup final model
cp -r output/ /workspace/backups/$(date +%Y%m%d_%H%M%S)/

echo "✅ Model backed up to /workspace/backups/"
```

### Download Model to Local Machine

```bash
# From your local machine (not RunPod):
# Replace POD_IP with your RunPod IP
scp -r root@POD_IP:/workspace/llm-finetune/output ./my-model-backup/
```

---

## Troubleshooting Cleanup

### "Permission Denied" Errors

```bash
# Run with sudo
sudo rm -rf output/

# Or change ownership
sudo chown -R $(whoami):$(whoami) /workspace/llm-finetune/
```

### "Directory Not Empty" Errors

```bash
# Force removal
rm -rf --no-preserve-root output/

# Or use find
find output/ -type f -delete
find output/ -type d -delete
```

### Files Won't Delete (In Use)

```bash
# Find processes using files
lsof +D /workspace/llm-finetune/output/

# Kill the processes
tmux kill-session -t training
pkill -f "python.*train"

# Try again
rm -rf output/
```

### Out of Disk Space

```bash
# Emergency cleanup (frees most space quickly)
rm -rf ~/.cache/huggingface/hub/models--*
rm -rf /workspace/llm-finetune/output/checkpoint-*
pip cache purge

# Check space
df -h /workspace
```

---

## Post-Cleanup Verification

### Verify Clean State

```bash
cd /workspace/llm-finetune

# Check for leftover files
ls -la

# Should NOT exist:
# - output/
# - processed_data/
# - config.yaml

# Should exist:
# - data/input/ (with your JSONL files)
# - scripts/
# - src/
# - configs/
```

### Start Fresh Training

```bash
# Pull latest code (if needed)
git pull

# Start training
bash scripts/runpod.sh train
```

---

## Quick Reference Commands

```bash
# Quick clean (most common)
cd /workspace/llm-finetune && rm -rf output/ processed_data/ config.yaml && tmux kill-session -t training 2>/dev/null; echo "✅ Clean"

# Update and clean
cd /workspace/llm-finetune && git pull && rm -rf output/ processed_data/ config.yaml && echo "✅ Updated"

# Nuclear reset
cd /workspace && rm -rf llm-finetune/ ~/.cache/huggingface/ && pip cache purge && echo "✅ Reset"

# Free max space
rm -rf ~/.cache/huggingface/ /workspace/llm-finetune/output/checkpoint-* && pip cache purge && echo "✅ Space freed"
```

---

## Best Practices

1. **Before Training:**
   - Always clean previous outputs: `rm -rf output/ processed_data/ config.yaml`
   - Verify your data: `ls -lh data/input/`
   - Pull latest code: `git pull`

2. **During Training:**
   - Monitor with: `tmux attach -t training`
   - Check GPU: `watch -n 1 nvidia-smi`
   - Check logs: `tail -f output/training.log`

3. **After Training:**
   - Backup your model before cleanup
   - Download to local machine
   - Clean up if done: `rm -rf output/`

4. **Disk Management:**
   - Keep only final checkpoint
   - Remove HF cache after training if space needed
   - Use separate volume for persistent storage

---

## Need Help?

- **Issue Tracker**: [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues)
- **Documentation**: [Full Docs](docs/README.md)
- **RunPod Guide**: [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md)
