# Phase 5: Model Deployment

This guide covers deploying your fine-tuned model for production use.

## Prerequisites

✅ Completed [PHASE4_EVALUATION.md](PHASE4_EVALUATION.md)
✅ Merged or exported model ready

## Deployment Options Overview

| Option | Best For | Cost | Complexity | Scalability |
|--------|----------|------|------------|-------------|
| Local (llama.cpp) | Personal/testing | Free | Low | Single user |
| RunPod Serverless | Auto-scaling API | Pay-per-use | Medium | High |
| HuggingFace Inference | Easy deployment | $$ | Low | Medium |
| vLLM | High throughput | Server cost | Medium | High |
| Custom API | Full control | Server cost | High | Custom |

## Option 1: Local Deployment with llama.cpp

Best for: Personal use, testing, CPU inference

### Step 1: Convert to GGUF (if not done)

```bash
python -m finetune_project export \
    --model /workspace/merged-model \
    --output /workspace/export \
    --format gguf
```

### Step 2: Download and Install llama.cpp

On your local machine:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build (Mac/Linux)
make

# Build (Windows with Visual Studio)
cmake -B build
cmake --build build --config Release
```

### Step 3: Download Model

```bash
# Download GGUF file from RunPod
scp root@<pod-ip>:/workspace/export/model-q4.gguf ./my-model-q4.gguf
```

### Step 4: Run Inference

```bash
# Interactive chat
./main -m my-model-q4.gguf -n 512 -c 2048 --color -i

# With specific prompt
./main -m my-model-q4.gguf -p "Explain quantum computing:" -n 256

# Server mode (API)
./server -m my-model-q4.gguf --host 0.0.0.0 --port 8080

# Test API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Option 2: RunPod Serverless Deployment

Best for: Auto-scaling API, pay-per-use

### Step 1: Create Serverless Endpoint

1. Go to RunPod → Serverless
2. Click "New Endpoint"
3. Select template: "Transformers vLLM"
4. Configure:
   - Model: Upload your model or use HF repo
   - Min workers: 0 (cost-effective)
   - Max workers: 5
   - GPU: A40 or L40

### Step 2: Deploy Model

```bash
# Upload model to HuggingFace (if not already)
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/workspace/merged-model',
    repo_id='your-username/your-model',
    repo_type='model'
)
"

# Or use RunPod's model upload feature
```

### Step 3: Call Endpoint

```python
import requests

endpoint_id = "your-endpoint-id"
api_key = "your-runpod-api-key"

response = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "input": {
            "prompt": "Explain quantum computing:",
            "max_tokens": 256,
            "temperature": 0.7
        }
    }
)

print(response.json())
```

## Option 3: HuggingFace Inference Endpoints

Best for: Easy deployment, managed infrastructure

### Step 1: Upload Model

```bash
# Login
huggingface-cli login

# Upload
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/workspace/merged-model',
    repo_id='your-username/your-model',
    repo_type='model',
    private=True  # or False for public
)
"
```

### Step 2: Create Inference Endpoint

1. Go to https://huggingface.co/your-username/your-model
2. Click "Deploy" → "Inference Endpoints"
3. Configure:
   - Instance: GPU (A10G, A100)
   - Auto-scaling: Min 0, Max 3
   - Region: US-East or EU-West

4. Click "Create Endpoint"

### Step 3: Call Endpoint

```python
import requests

API_URL = "https://your-endpoint.endpoints.huggingface.cloud"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Explain quantum computing:",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7
    }
})

print(output)
```

## Option 4: vLLM Self-Hosted

Best for: High throughput, batch processing

### Step 1: Install vLLM

```bash
pip install vllm
```

### Step 2: Create Server Script

```python
# server.py
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
llm = LLM(
    model="/workspace/merged-model",
    tensor_parallel_size=1,  # Number of GPUs
    dtype="bfloat16"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/generate")
def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    outputs = llm.generate([request.prompt], sampling_params)
    return {"output": outputs[0].outputs[0].text}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Step 3: Start Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Step 4: Call API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Option 5: Custom FastAPI Deployment

Best for: Full control, custom logic

### Step 1: Create API Server

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager

# Global model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("/workspace/merged-model")
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/merged-model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded!")
    yield
    # Cleanup on shutdown
    del model, tokenizer

app = FastAPI(lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "prompt": request.prompt,
            "output": response,
            "tokens": outputs.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Step 2: Create Dockerfile

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy model and code
COPY merged-model /app/merged-model
COPY api_server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 3: Build and Run

```bash
# Build image
docker build -t my-llm-api .

# Run container
docker run -d \
    --gpus all \
    -p 8000:8000 \
    --name llm-api \
    my-llm-api

# Test
curl http://localhost:8000/health
```

## Production Considerations

### 1. Load Balancing

```nginx
# nginx.conf
upstream llm_backend {
    server 10.0.0.1:8000;
    server 10.0.0.2:8000;
    server 10.0.0.3:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Rate Limiting

```python
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate(request: Request, gen_request: GenerateRequest):
    # Your generation code
    pass
```

### 3. Monitoring

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.post("/generate")
async def generate(request: GenerateRequest):
    request_count.inc()
    with request_duration.time():
        # Your generation code
        pass

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 4. Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_generate(prompt_hash, max_tokens, temperature):
    # Generation logic
    pass

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
    return cached_generate(prompt_hash, request.max_tokens, request.temperature)
```

## Cost Optimization

### Cloud GPU Costs (Monthly, 24/7):

| Provider | GPU | Cost/hr | Monthly |
|----------|-----|---------|---------|
| RunPod | RTX 4090 | $0.44 | ~$317 |
| RunPod | A40 | $0.39 | ~$281 |
| RunPod | A100 40GB | $1.89 | ~$1,361 |
| AWS | g5.xlarge (A10G) | $1.01 | ~$727 |
| GCP | g2-standard-4 (L4) | $0.90 | ~$648 |

### Cost Reduction Strategies:

1. **Serverless** - Only pay when used
2. **Spot Instances** - 70% cheaper but interruptible
3. **Quantization** - Q4 models use less VRAM, cheaper GPUs
4. **Batching** - Process multiple requests together
5. **Auto-scaling** - Scale down during low traffic

## Deployment Checklist

- [ ] Model tested and working
- [ ] API endpoints secured (auth tokens)
- [ ] Rate limiting implemented
- [ ] Monitoring/logging configured
- [ ] Auto-scaling enabled
- [ ] Backup strategy in place
- [ ] Cost alerts configured
- [ ] Documentation for users

## Troubleshooting

### High latency

1. Enable Flash Attention
2. Use vLLM instead of vanilla transformers
3. Reduce max_tokens
4. Use continuous batching
5. Quantize model to Q4/Q8

### Out of memory in production

1. Reduce max_seq_length
2. Use smaller model or quantization
3. Enable KV cache quantization
4. Limit concurrent requests

### Model not loading

```bash
# Check model files
ls -lh /workspace/merged-model/

# Verify model config
cat /workspace/merged-model/config.json

# Test loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/workspace/merged-model')
print('Model loaded successfully!')
"
```

## Next Steps

🎉 **Congratulations!** Your model is now deployed and serving requests!

For ongoing maintenance:
- Monitor performance metrics
- Collect user feedback
- Consider iterative improvements
- Plan for model updates

## Quick Reference

```bash
# llama.cpp server
./server -m model.gguf --host 0.0.0.0 --port 8080

# vLLM server
python -m vllm.entrypoints.openai.api_server --model /workspace/merged-model

# FastAPI server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

# Docker deployment
docker run -d --gpus all -p 8000:8000 my-llm-api

# Health check
curl http://localhost:8000/health

# Test generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```
