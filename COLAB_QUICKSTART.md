# Google Colab Quick Start

The **easiest and fastest** way to fine-tune your LLM - no setup required!

## 🚀 Start Training in 5 Minutes

### Step 1: Open Notebook

Click this badge to open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ravidsun/llm-finetune/blob/master/notebooks/llm_finetune_colab.ipynb)

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 for free, A100 with Pro+)
3. Click **Save**

### Step 3: Run Cells

Simply run each cell in order (Shift+Enter or click ▶️)

The notebook will:
- ✅ Install all dependencies (~3 minutes)
- ✅ Clone this repository
- ✅ Set up your environment
- ✅ Prepare your data
- ✅ Train your model
- ✅ Test the results
- ✅ Save to Google Drive

### Step 4: Download Model

After training completes, download your fine-tuned model or upload to Hugging Face!

## 📊 What You Get

**Free Tier (Colab Free)**
- GPU: Tesla T4 (15GB VRAM)
- Perfect for 7B models with QLoRA
- Training time: ~30-60 min (1K samples)
- Cost: **FREE**

**Pro+ Tier (Colab Pro+)**
- GPU: V100/A100 (up to 40GB VRAM)
- Great for 7B-14B models
- Training time: ~15-30 min (1K samples)
- Cost: $50/month

## 📖 Documentation

- **[Complete Colab Guide](docs/COLAB_GUIDE.md)** - Detailed instructions
- **[Jupyter Notebook](notebooks/llm_finetune_colab.ipynb)** - Ready-to-run notebook
- **[Standalone Script](scripts/colab_finetune.py)** - Python script version

## 🎯 Perfect For

- ✅ Beginners to LLM fine-tuning
- ✅ Quick experiments and testing
- ✅ No local GPU available
- ✅ Learning and education
- ✅ Small to medium datasets (1K-10K samples)

## ⚠️ Limitations

- **Session timeout**: 12 hours (free), 24 hours (Pro)
- **Idle disconnect**: ~90 minutes of inactivity
- **No persistence**: Save to Google Drive!
- **Model size**: Best for 7B models on free tier

## 💡 Tips

1. **Mount Google Drive** to save your model automatically
2. **Save frequently** - sessions can disconnect
3. **Use sample data first** to test the pipeline
4. **Monitor VRAM usage** with `!nvidia-smi`
5. **Download immediately** after training completes

## 🆚 Colab vs RunPod

| Feature | Colab Free | Colab Pro+ | RunPod |
|---------|------------|------------|--------|
| GPU | T4 (15GB) | A100 (40GB) | Custom |
| Cost | Free | $50/month | Pay-per-hour |
| Session | 12 hours | 24 hours | Unlimited |
| Setup | Zero | Zero | Some |
| Best For | Learning | Medium models | Production |

## 🔗 Quick Links

- [Open Colab Notebook](https://colab.research.google.com/github/ravidsun/llm-finetune/blob/master/notebooks/llm_finetune_colab.ipynb)
- [Full Documentation](docs/README.md)
- [GitHub Repository](https://github.com/ravidsun/llm-finetune)

---

**Ready to start?** Click the badge at the top and begin fine-tuning! 🚀
