#!/usr/bin/env python3
"""
Standalone script for fine-tuning LLMs on Google Colab
Can be run directly without installing the full package

Usage:
    python colab_finetune.py --model Qwen/Qwen2.5-7B-Instruct --data data.jsonl
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Check if running on Colab
try:
    import google.colab
    ON_COLAB = True
except:
    ON_COLAB = False


def check_gpu():
    """Check GPU availability"""
    if not torch.cuda.is_available():
        print("⚠️ WARNING: No GPU detected!")
        print("Go to Runtime → Change runtime type → GPU")
        sys.exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    os.system("pip install -q -U transformers datasets accelerate peft trl bitsandbytes")
    print("✅ Dependencies installed!")


def load_dataset_from_jsonl(file_path: str) -> List[Dict]:
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_instruction(example: Dict) -> str:
    """Format example as instruction-following prompt"""
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get('input'):
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return text


def train_model(
    model_name: str,
    data_file: str,
    output_dir: str = "output",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
):
    """
    Train model with LoRA/QLoRA

    Args:
        model_name: HuggingFace model name
        data_file: Path to JSONL training data
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        use_4bit: Use 4-bit quantization (QLoRA)
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"Starting Fine-Tuning")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Data: {data_file}")
    print(f"Output: {output_dir}")
    print(f"4-bit: {use_4bit}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print("Loading model...")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # LoRA configuration
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}%")

    # Load and format dataset
    print(f"\nLoading dataset from {data_file}...")
    data = load_dataset_from_jsonl(data_file)
    print(f"Loaded {len(data)} examples")

    # Format data
    formatted_data = [{"text": format_instruction(ex)} for ex in data]
    dataset = Dataset.from_list(formatted_data)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
        report_to=[],
    )

    # Trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting Training ({num_epochs} epochs)")
    print(f"{'='*60}\n")

    trainer.train()

    # Save model
    print(f"\n{'='*60}")
    print(f"Training Complete! Saving model...")
    print(f"{'='*60}\n")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model saved to: {output_dir}")
    print(f"✅ Fine-tuning complete!")

    return model, tokenizer


def test_model(model, tokenizer, prompts: List[str]):
    """Test the trained model"""
    print(f"\n{'='*60}")
    print(f"Testing Model")
    print(f"{'='*60}\n")

    for prompt in prompts:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()

        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on Google Colab")

    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name from HuggingFace")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data (JSONL)")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")

    # Other
    parser.add_argument("--test", action="store_true",
                       help="Test model after training")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")

    args = parser.parse_args()

    # Check GPU
    check_gpu()

    # Install dependencies
    if not args.skip_install:
        install_dependencies()

    # Train
    model, tokenizer = train_model(
        model_name=args.model,
        data_file=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
    )

    # Test
    if args.test:
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?"
        ]
        test_model(model, tokenizer, test_prompts)

    # Save to Drive if on Colab
    if ON_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)

            drive_path = f"/content/drive/MyDrive/llm-finetune/{args.output}"
            os.system(f"mkdir -p {drive_path}")
            os.system(f"cp -r {args.output}/* {drive_path}/")
            print(f"\n✅ Model also saved to Google Drive: {drive_path}")
        except:
            print("\nℹ️  Mount Google Drive manually to save model persistently")

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
