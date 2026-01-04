# How to Download Output from RunPod

Quick guide for downloading your trained model adapter from RunPod to your local Windows machine.

## Prerequisites

- Completed training on RunPod
- Training output exists at `/workspace/llm-finetune/output/` on RunPod

## Method 1: RunPod Web Interface (Easiest)

### Step 1: Access RunPod File Browser

1. Log into [RunPod](https://www.runpod.io/)
2. Go to "My Pods"
3. Find your pod and click **"Connect"**
4. Click **"Connect to HTTP Service [Port 8888]"** or similar

### Step 2: Navigate to Output Folder

1. In the file browser, navigate to:
   ```
   /workspace/llm-finetune/output/
   ```

2. You should see files like:
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - Other tokenizer files

### Step 3: Download the Folder

**Option A: Download as ZIP (Recommended)**
1. Right-click on the `output` folder
2. Select "Download as ZIP" or "Download"
3. Save to your Downloads folder
4. Extract the ZIP file to `c:\LLM\llm-finetune\output\`

**Option B: Download Individual Files**
1. Select all files in the `output` folder
2. Download them
3. Place in `c:\LLM\llm-finetune\output\` on your local machine

### Step 4: Verify Download

Check that you have the files:

```cmd
dir c:\LLM\llm-finetune\output
```

You should see at minimum:
- `adapter_config.json` ✅
- `adapter_model.safetensors` (or `adapter_model.bin`) ✅

## Method 2: SCP (Command Line)

### Step 1: Get Your RunPod IP

1. In RunPod dashboard, find your pod
2. Click "Connect"
3. Look for the **SSH command** or **IP address**
4. Copy the IP (e.g., `123.45.67.89`)

### Step 2: Download via SCP

Open Command Prompt or PowerShell on your local machine:

```cmd
# Replace POD_IP with your actual RunPod IP
scp -r root@POD_IP:/workspace/llm-finetune/output c:\LLM\llm-finetune\

# Example:
# scp -r root@123.45.67.89:/workspace/llm-finetune/output c:\LLM\llm-finetune\
```

**If asked for password:**
- Default RunPod password is usually shown in the dashboard
- Or use SSH key if you configured one

### Step 3: Verify Download

```cmd
dir c:\LLM\llm-finetune\output
```

## Method 3: Using RunPod CLI

### Step 1: Install RunPod CLI

```cmd
pip install runpodctl
```

### Step 2: Login

```cmd
runpodctl config --apiKey YOUR_API_KEY
```

### Step 3: Download Files

```cmd
# List your pods
runpodctl get pods

# Download from specific pod
runpodctl receive POD_ID:/workspace/llm-finetune/output c:\LLM\llm-finetune\output
```

## Method 4: Using WinSCP (GUI Alternative)

### Step 1: Download WinSCP

1. Download from [winscp.net](https://winscp.net/)
2. Install WinSCP

### Step 2: Connect to RunPod

1. Open WinSCP
2. Enter connection details:
   - **Protocol:** SFTP
   - **Host:** Your RunPod IP
   - **Port:** 22
   - **Username:** root
   - **Password:** (from RunPod dashboard)

3. Click "Login"

### Step 3: Navigate and Download

1. In the remote panel (right side), navigate to:
   ```
   /workspace/llm-finetune/output/
   ```

2. In the local panel (left side), navigate to:
   ```
   c:\LLM\llm-finetune\
   ```

3. Select the `output` folder on the right
4. Drag it to the left panel
5. Wait for transfer to complete

## Verify Your Download

After downloading, verify you have the correct files:

```cmd
cd c:\LLM\llm-finetune
dir output

# Check for adapter files
dir output\adapter_config.json
dir output\adapter_model.safetensors
```

**Expected output:**
```
Directory of c:\LLM\llm-finetune\output

adapter_config.json
adapter_model.safetensors (or .bin)
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
added_tokens.json
training_args.bin
README.md
...
```

## File Sizes (Approximate)

- **adapter_model.safetensors:** 200-500 MB (LoRA weights)
- **adapter_config.json:** Few KB
- **Tokenizer files:** Few MB total
- **Total download:** ~300-600 MB

## Troubleshooting

### "Connection refused" or "Connection timeout"

**Solution:**
1. Make sure your RunPod pod is running
2. Check the IP address is correct
3. Try using the web interface instead

### "Permission denied"

**Solution:**
1. Check you're using the correct password
2. Make sure you're logged in as `root` user

### Download is very slow

**Solution:**
1. RunPod's network speed varies by region
2. Try downloading during off-peak hours
3. Use the web interface ZIP download instead

### Files are incomplete or corrupted

**Solution:**
```cmd
# Re-download the files
# Then verify sizes match what you see on RunPod:
dir /s c:\LLM\llm-finetune\output
```

## After Download

Once you have the files downloaded:

1. **Verify location:**
   ```cmd
   dir c:\LLM\llm-finetune\output\adapter_config.json
   ```

2. **Run merge:**
   ```cmd
   cd c:\LLM\llm-finetune
   python scripts/merge_local.py
   ```

3. **See full workflow:**
   - [MERGE_WORKFLOW.md](MERGE_WORKFLOW.md) for complete merge instructions

## Clean Up RunPod (Optional)

After successful download, you can:

1. **Stop the pod** (to save money)
2. **Delete the pod** (if you're completely done)
3. **Keep the pod** (if you want to do more training)

**To verify files before deleting:**
```bash
# On RunPod, check what you have:
ls -lh /workspace/llm-finetune/output/

# Make sure you downloaded everything!
```

## Quick Reference

**Web Interface:**
1. RunPod → Connect → HTTP Service
2. Navigate to `/workspace/llm-finetune/output/`
3. Download as ZIP
4. Extract to `c:\LLM\llm-finetune\output\`

**SCP Command:**
```cmd
scp -r root@POD_IP:/workspace/llm-finetune/output c:\LLM\llm-finetune\
```

**Verify:**
```cmd
dir c:\LLM\llm-finetune\output\adapter_config.json
```

**Next Step:**
```cmd
python scripts/merge_local.py
```

## Need Help?

- [MERGE_WORKFLOW.md](MERGE_WORKFLOW.md) - Complete merge workflow
- [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) - RunPod usage guide
- [GitHub Issues](https://github.com/ravidsun/llm-finetune/issues) - Report problems
