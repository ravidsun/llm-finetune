#!/usr/bin/env python3
"""
Test Merged Model
=================
Quick test script to verify your merged model works correctly.

Usage:
    python scripts/test_merged_model.py /workspace/llm-finetune/merged_model
    python scripts/test_merged_model.py /workspace/llm-finetune/merged_model --prompt "Your test prompt"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test merged model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to merged model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to test (optional)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    return parser.parse_args()


def test_model(args):
    print("=" * 70)
    print("🧪 Testing Merged Model")
    print("=" * 70)
    print()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"📍 Model path: {args.model_path}")
    print(f"💻 Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
    print()

    # Load model and tokenizer
    print("⏳ Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    print()

    # Prepare test prompts
    if args.prompt:
        test_prompts = [args.prompt]
    else:
        # Default test prompts - adjust based on your training data
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
        ]

    print(f"🎯 Running {len(test_prompts)} test prompt(s)...")
    print()

    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print("-" * 70)
        print(f"Test {i}/{len(test_prompts)}")
        print("-" * 70)
        print(f"📝 Prompt: {prompt}")
        print()

        # Format as chat if the model expects it
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback to plain text
            text = prompt

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        print("⏳ Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part (remove prompt)
        if text in response:
            generated = response[len(text):].strip()
        else:
            generated = response.strip()

        print()
        print(f"🤖 Response:")
        print(generated)
        print()

    print("=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("   - Adjust --temperature (0.1-1.0) for different creativity levels")
    print("   - Use --max_length to control response length")
    print("   - Test with your specific use case prompts")
    print()


def main():
    args = parse_args()

    # Verify model path exists
    import os
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        return

    # Check for model files
    required_files = ["config.json", "tokenizer_config.json"]
    has_model = any(
        os.path.exists(os.path.join(args.model_path, f))
        for f in ["model.safetensors", "pytorch_model.bin"]
    )

    if not has_model:
        print(f"❌ Error: No model weights found in {args.model_path}")
        print("   Looking for: model.safetensors or pytorch_model.bin")
        return

    # Run test
    test_model(args)


if __name__ == "__main__":
    main()
