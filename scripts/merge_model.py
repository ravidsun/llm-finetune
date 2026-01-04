#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model
===================================
This script merges your trained LoRA adapter with the base model to create
a standalone merged model that can be used for inference without loading adapters.

Usage:
    python scripts/merge_model.py --adapter_path /workspace/llm-finetune/output \
                                   --output_path /workspace/llm-finetune/merged_model \
                                   --base_model Qwen/Qwen2.5-7B-Instruct

Requirements:
    - Sufficient disk space (2x model size, ~30-60GB for 7B/14B models)
    - VRAM not required (merging happens on CPU)
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained adapter (e.g., /workspace/llm-finetune/output)",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model (e.g., /workspace/llm-finetune/merged_model)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for merging (default: auto)",
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum size of each model shard (default: 5GB)",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace Hub model ID (required if --push_to_hub is set)",
    )

    return parser.parse_args()


def verify_adapter_exists(adapter_path: str) -> bool:
    """Verify that the adapter files exist."""
    adapter_path = Path(adapter_path)

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"  # or adapter_model.bin
    ]

    # Check for safetensors or bin format
    has_safetensors = (adapter_path / "adapter_model.safetensors").exists()
    has_bin = (adapter_path / "adapter_model.bin").exists()

    if not has_safetensors and not has_bin:
        print(f"❌ Error: No adapter weights found in {adapter_path}")
        print(f"   Looking for: adapter_model.safetensors or adapter_model.bin")
        return False

    if not (adapter_path / "adapter_config.json").exists():
        print(f"❌ Error: adapter_config.json not found in {adapter_path}")
        return False

    print(f"✅ Adapter files verified in {adapter_path}")
    print(f"   Using: {'adapter_model.safetensors' if has_safetensors else 'adapter_model.bin'}")

    return True


def get_device(device_arg: str) -> str:
    """Determine the device to use."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def merge_model(args):
    """Main merging function."""

    print("=" * 70)
    print("🔄 LoRA Adapter + Base Model → Merged Model")
    print("=" * 70)
    print()

    # Step 1: Verify adapter exists
    print("[1/6] Verifying adapter files...")
    if not verify_adapter_exists(args.adapter_path):
        sys.exit(1)
    print()

    # Step 2: Determine device
    device = get_device(args.device)
    print(f"[2/6] Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print()

    # Step 3: Load base model
    print(f"[3/6] Loading base model: {args.base_model}")
    print("   This may take several minutes...")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        print(f"✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        sys.exit(1)
    print()

    # Step 4: Load adapter
    print(f"[4/6] Loading LoRA adapter from: {args.adapter_path}")

    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            device_map=device if device != "cpu" else None,
        )
        print(f"✅ Adapter loaded successfully")
    except Exception as e:
        print(f"❌ Error loading adapter: {e}")
        sys.exit(1)
    print()

    # Step 5: Merge adapter with base model
    print("[5/6] Merging adapter with base model...")
    print("   This may take several minutes...")

    try:
        merged_model = model.merge_and_unload()
        print(f"✅ Model merged successfully")
    except Exception as e:
        print(f"❌ Error merging model: {e}")
        sys.exit(1)
    print()

    # Step 6: Save merged model
    print(f"[6/6] Saving merged model to: {args.output_path}")

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save the merged model
        merged_model.save_pretrained(
            args.output_path,
            max_shard_size=args.max_shard_size,
            safe_serialization=True,  # Use safetensors format
        )
        print(f"✅ Model saved successfully")

        # Load and save tokenizer
        print("   Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(args.output_path)
        print(f"✅ Tokenizer saved successfully")

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        sys.exit(1)
    print()

    # Optional: Push to Hub
    if args.push_to_hub:
        if not args.hub_model_id:
            print("❌ Error: --hub_model_id required when using --push_to_hub")
            sys.exit(1)

        print(f"📤 Pushing to HuggingFace Hub: {args.hub_model_id}")
        try:
            merged_model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
            print(f"✅ Model pushed to Hub successfully")
        except Exception as e:
            print(f"❌ Error pushing to Hub: {e}")
            sys.exit(1)
        print()

    # Summary
    print("=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print()
    print(f"📁 Merged model location: {args.output_path}")
    print()
    print("📊 Model files:")
    for file in sorted(output_path.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"   - {file.name} ({size_mb:.1f} MB)")
    print()
    print("🚀 Usage:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   ")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.output_path}')")
    print()

    # Calculate total size
    total_size_gb = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file()) / 1024**3
    print(f"💾 Total size: {total_size_gb:.2f} GB")
    print()


def main():
    args = parse_args()

    # Verify paths
    if not os.path.exists(args.adapter_path):
        print(f"❌ Error: Adapter path does not exist: {args.adapter_path}")
        sys.exit(1)

    # Check disk space
    output_parent = Path(args.output_path).parent
    if output_parent.exists():
        stat = os.statvfs(output_parent)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        print(f"💾 Available disk space: {free_space_gb:.1f} GB")
        if free_space_gb < 30:
            print(f"⚠️  Warning: Low disk space. Merging may require 30-60 GB.")
            response = input("Continue anyway? (y/N): ")
            if not response.lower().startswith('y'):
                sys.exit(0)
        print()

    # Run merge
    merge_model(args)


if __name__ == "__main__":
    main()
