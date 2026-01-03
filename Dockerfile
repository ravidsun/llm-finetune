# Dockerfile for LLM Fine-Tuning with LangChain on RunPod
# Based on PyTorch with CUDA support

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch ecosystem
RUN pip install --upgrade torch torchvision torchaudio

# Install Hugging Face ecosystem
RUN pip install --upgrade \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.27.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    safetensors>=0.4.0

# Install bitsandbytes for quantization
RUN pip install --upgrade bitsandbytes>=0.43.0

# Install Flash Attention 2
RUN pip install flash-attn --no-build-isolation || echo "Flash attention installation skipped"

# Install LangChain ecosystem
RUN pip install --upgrade \
    langchain>=0.2.0 \
    langchain-core>=0.2.0 \
    langchain-community>=0.2.0 \
    langchain-text-splitters>=0.2.0

# Install optional LLM providers for QA generation
RUN pip install --upgrade \
    langchain-anthropic>=0.1.0 \
    anthropic>=0.25.0 \
    || echo "Anthropic installation skipped"

# Install document processing libraries
RUN pip install --upgrade \
    pymupdf>=1.24.0 \
    pdfplumber>=0.10.0 \
    unstructured>=0.12.0 \
    python-docx>=1.1.0

# Install CLI and utility dependencies
RUN pip install --upgrade \
    typer[all]>=0.9.0 \
    rich>=13.0.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0

# Install experiment tracking
RUN pip install --upgrade wandb>=0.16.0

# Install additional utilities
RUN pip install --upgrade \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    tiktoken>=0.6.0 \
    einops>=0.7.0 \
    scipy>=1.11.0 \
    jinja2>=3.1.0

# Create workspace directories
RUN mkdir -p /workspace/data /workspace/output /workspace/hf_cache

# Copy project files
WORKDIR /app
COPY . /app/

# Install the project
RUN pip install -e .

# Set default command
CMD ["/bin/bash"]
