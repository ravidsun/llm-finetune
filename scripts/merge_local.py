#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model - Local Windows Version
===========================================================
This script merges your trained LoRA adapter with the base model to create
a standalone merged model. Optimized for local Windows environments.

Usage:
    python scripts/merge_local.py

    # Or with custom paths:
    python scripts/merge_local.py --adapter_path ./output --output_path ./merged_model

Requirements:
    - Sufficient disk space (2x model size, ~30-60GB for 7B/14B models)
    - Python packages: transformers, peft, torch, accelerate, safetensors
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Check dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install transformers peft torch accelerate safetensors")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model (Local Windows)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./output",
        help="Path to the trained adapter (default: ./output)",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./merged_model",
        help="Path to save the merged model (default: ./merged_model)",
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

    return parser.parse_args()


def verify_adapter_exists(adapter_path: Path) -> bool:
    """Verify that the adapter files exist."""
    print(f"\n[1/7] Checking adapter files...")
    print(f"   Path: {adapter_path.absolute()}")

    if not adapter_path.exists():
        print(f"❌ Error: Adapter path does not exist: {adapter_path}")
        return False

    required_files = ["adapter_config.json"]
    has_safetensors = (adapter_path / "adapter_model.safetensors").exists()
    has_bin = (adapter_path / "adapter_model.bin").exists()

    if not has_safetensors and not has_bin:
        print(f"❌ Error: No adapter weights found in {adapter_path}")
        print(f"   Looking for: adapter_model.safetensors or adapter_model.bin")
        return False

    if not (adapter_path / "adapter_config.json").exists():
        print(f"❌ Error: adapter_config.json not found in {adapter_path}")
        return False

    print(f"✅ Adapter files verified")
    print(f"   Using: {'adapter_model.safetensors' if has_safetensors else 'adapter_model.bin'}")

    return True


def auto_detect_base_model(adapter_path: Path) -> str:
    """Auto-detect base model from adapter config."""
    print(f"\n[2/7] Auto-detecting base model...")

    try:
        config_file = adapter_path / "adapter_config.json"
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        base_model = config.get("base_model_name_or_path", "")
        if base_model:
            print(f"✅ Detected base model: {base_model}")
            return base_model
    except Exception as e:
        print(f"⚠️  Could not read adapter config: {e}")

    # Try to read from training config
    config_yaml = Path("config.yaml")
    if config_yaml.exists():
        try:
            import yaml
            with open(config_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            base_model = config.get("model", {}).get("model_name", "")
            if base_model:
                print(f"✅ Found in config.yaml: {base_model}")
                return base_model
        except:
            pass

    return None


def prompt_for_base_model() -> str:
    """Prompt user to select base model."""
    print("\n⚠️  Could not auto-detect base model.")
    print("\nWhich base model did you use for training?")
    print("  1) Qwen/Qwen2.5-7B-Instruct")
    print("  2) Qwen/Qwen2.5-14B-Instruct")
    print("  3) Other (enter manually)")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            return "Qwen/Qwen2.5-7B-Instruct"
        elif choice == "2":
            return "Qwen/Qwen2.5-14B-Instruct"
        elif choice == "3":
            model = input("Enter base model name: ").strip()
            if model:
                return model

        print("❌ Invalid choice, please try again")


def check_disk_space(output_path: Path) -> bool:
    """Check if there's enough disk space."""
    print(f"\n[3/7] Checking disk space...")

    try:
        import shutil
        stat = shutil.disk_usage(output_path.parent if output_path.parent.exists() else Path.cwd())
        free_gb = stat.free / (1024**3)

        print(f"   Available space: {free_gb:.1f} GB")

        if free_gb < 30:
            print(f"⚠️  Warning: Low disk space! Merging may require 30-60 GB.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            return response in ['y', 'yes']

        print(f"✅ Sufficient disk space available")
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True


def get_device(device_arg: str) -> str:
    """Determine the device to use."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def merge_model(args):
    """Main merging function."""

    print("=" * 70)
    print("🔄 Merge LoRA Adapter with Base Model (Local)")
    print("=" * 70)

    # Convert paths
    adapter_path = Path(args.adapter_path).absolute()
    output_path = Path(args.output_path).absolute()

    # Step 1: Verify adapter
    if not verify_adapter_exists(adapter_path):
        return False

    # Step 2: Detect or prompt for base model
    base_model = args.base_model
    if not base_model:
        base_model = auto_detect_base_model(adapter_path)

    if not base_model:
        base_model = prompt_for_base_model()

    # Step 3: Check disk space
    if not check_disk_space(output_path):
        print("\n❌ Aborting due to insufficient disk space")
        return False

    # Step 4: Determine device
    device = get_device(args.device)
    print(f"\n[4/7] Device selection: {device}")
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        except:
            print("   GPU info unavailable")
    else:
        print("   Using CPU (this will be slower)")

    # Step 5: Load base model
    print(f"\n[5/7] Loading base model: {base_model}")
    print("   This may take several minutes...")

    try:
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        print("\nPossible solutions:")
        print("  1. Check internet connection (model downloads from HuggingFace)")
        print("  2. Verify base model name is correct")
        print("  3. Try running with --device cpu if GPU memory is insufficient")
        return False

    # Step 6: Load adapter
    print(f"\n[6/7] Loading LoRA adapter...")

    try:
        model = PeftModel.from_pretrained(
            base_model_obj,
            str(adapter_path),
            device_map=device if device != "cpu" else None,
        )
        print(f"✅ Adapter loaded successfully")
    except Exception as e:
        print(f"❌ Error loading adapter: {e}")
        return False

    # Merge adapter with base model
    print("   Merging adapter with base model...")
    print("   This may take several minutes...")

    try:
        merged_model = model.merge_and_unload()
        print(f"✅ Model merged successfully")
    except Exception as e:
        print(f"❌ Error merging model: {e}")
        return False

    # Step 7: Save merged model
    print(f"\n[7/7] Saving merged model...")
    print(f"   Output: {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save the merged model
        print("   Saving model weights...")
        merged_model.save_pretrained(
            str(output_path),
            max_shard_size=args.max_shard_size,
            safe_serialization=True,  # Use safetensors format
        )
        print(f"✅ Model saved successfully")

        # Load and save tokenizer
        print("   Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(str(output_path))
        print(f"✅ Tokenizer saved successfully")

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Merged model location:")
    print(f"   {output_path}")
    print()
    print("📊 Model files:")
    total_size = 0
    for file in sorted(output_path.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += file.stat().st_size
            print(f"   - {file.name} ({size_mb:.1f} MB)")

    total_size_gb = total_size / 1024**3
    print(f"\n💾 Total size: {total_size_gb:.2f} GB")

    print()
    print("🧪 Test your merged model:")
    print(f"   python scripts/test_merged_model.py \"{output_path}\"")
    print()
    print("🚀 Use in Python:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   model = AutoModelForCausalLM.from_pretrained(\"{output_path}\")")
    print(f"   tokenizer = AutoTokenizer.from_pretrained(\"{output_path}\")")
    print()

    return True


def main():
    print("=" * 70)
    print("Local Model Merge Tool")
    print("=" * 70)

    args = parse_args()

    # Run merge
    success = merge_model(args)

    if success:
        print("✅ All done!")
        sys.exit(0)
    else:
        print("\n❌ Merge failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
