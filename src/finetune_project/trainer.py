"""
Training pipeline for fine-tuning LLMs with PEFT/LoRA support.

Uses TRL's SFTTrainer for chat models with proper prompt formatting.
Integrates with LangChain data pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict

from finetune_project.config import Config

logger = logging.getLogger(__name__)


class FineTuneTrainer:
    """
    Fine-tuning trainer with PEFT/LoRA and quantization support.
    
    Uses TRL's SFTTrainer which is optimized for instruction fine-tuning
    with proper chat template handling and packing support.
    
    Why TRL SFTTrainer over base Trainer:
    - Native support for chat templates and conversation formatting
    - Built-in sequence packing for efficient training
    - Designed specifically for supervised fine-tuning
    - Better integration with PEFT adapters
    """

    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Full configuration object.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_format = None
        self._setup_environment()

    def _setup_environment(self):
        """Configure environment variables and settings."""
        os.environ["HF_HOME"] = self.config.runpod.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.config.runpod.hf_cache_dir
        
        self._set_seed(self.config.training.seed)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_quantization_config(self):
        """Build BitsAndBytes quantization config if enabled."""
        quant_config = self.config.quantization
        
        if not quant_config.enabled:
            return None
        
        from transformers import BitsAndBytesConfig
        
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            quant_config.bnb_4bit_compute_dtype,
            torch.bfloat16
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=quant_config.load_in_4bit,
            load_in_8bit=quant_config.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )

    def _get_peft_config(self):
        """Build PEFT/LoRA configuration if enabled."""
        peft_config = self.config.peft
        
        if not peft_config.enabled:
            return None
        
        from peft import LoraConfig, TaskType
        
        target_modules = peft_config.target_modules
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        
        return LoraConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            task_type=task_type_map.get(peft_config.task_type, TaskType.CAUSAL_LM),
            target_modules=target_modules,
            modules_to_save=peft_config.modules_to_save,
        )

    def _load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_config = self.config.model
        
        logger.info(f"Loading model: {model_config.model_name_or_path}")
        
        torch_dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = torch_dtype_map.get(model_config.torch_dtype, "auto")
        
        model_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        
        quantization_config = self._get_quantization_config()
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        
        if model_config.attn_implementation:
            model_kwargs["attn_implementation"] = model_config.attn_implementation
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs
        )
        
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        
        tokenizer_name = model_config.tokenizer_name or model_config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
        if self.config.peft.enabled:
            self._apply_peft()

    def _apply_peft(self):
        """Apply PEFT/LoRA adapters to the model."""
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        logger.info("Applying PEFT/LoRA configuration...")
        
        if self.config.quantization.enabled:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )
        
        peft_config = self._get_peft_config()
        self.model = get_peft_model(self.model, peft_config)
        
        trainable_params, all_params = self._count_parameters()
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    def _count_parameters(self):
        """Count trainable and total parameters."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, all_params

    def _get_training_arguments(self):
        """Build training arguments."""
        from transformers import TrainingArguments
        
        train_config = self.config.training
        
        fp16 = train_config.fp16
        bf16 = train_config.bf16
        
        if not fp16 and not bf16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                bf16 = True
            else:
                fp16 = True
        
        return TrainingArguments(
            output_dir=train_config.output_dir,
            num_train_epochs=train_config.num_train_epochs,
            max_steps=train_config.max_steps if train_config.max_steps > 0 else -1,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            learning_rate=train_config.learning_rate,
            warmup_ratio=train_config.warmup_ratio,
            lr_scheduler_type=train_config.lr_scheduler_type,
            weight_decay=train_config.weight_decay,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=train_config.gradient_checkpointing,
            logging_steps=train_config.logging_steps,
            save_steps=train_config.save_steps,
            eval_steps=train_config.eval_steps,
            save_total_limit=train_config.save_total_limit,
            eval_strategy=train_config.eval_strategy,  # Renamed from evaluation_strategy
            optim=train_config.optim,
            max_grad_norm=train_config.max_grad_norm,
            group_by_length=train_config.group_by_length,
            report_to=train_config.report_to,
            dataloader_num_workers=train_config.dataloader_num_workers,
            remove_unused_columns=train_config.remove_unused_columns,
            seed=train_config.seed,
            data_seed=train_config.seed,
        )

    def _format_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Format dataset for training."""
        train_data = dataset["train"]
        sample = train_data[0]
        
        if "messages" in sample and sample["messages"]:
            logger.info("Detected chat/messages format - using chat template")
            self.data_format = "chat"
        elif "text" in sample and sample["text"]:
            logger.info("Detected text format - using raw text for causal LM")
            self.data_format = "text"
        else:
            raise ValueError("Dataset must have either 'messages' or 'text' field")
        
        return dataset

    def train(self, dataset: DatasetDict):
        """
        Run the training loop.
        
        Args:
            dataset: DatasetDict with train and optionally eval splits.
        """
        from trl import SFTTrainer, SFTConfig
        
        self._load_model_and_tokenizer()
        
        dataset = self._format_dataset(dataset)
        
        training_args = self._get_training_arguments()
        
        sft_config = SFTConfig(
            **training_args.to_dict(),
            max_seq_length=self.config.training.max_seq_length,
            packing=self.config.training.packing and self.data_format == "text",
            dataset_text_field="text" if self.data_format == "text" else None,
        )
        
        trainer_kwargs = {
            "model": self.model,
            "args": sft_config,
            "train_dataset": dataset["train"],
            "tokenizer": self.tokenizer,
        }
        
        if "eval" in dataset:
            trainer_kwargs["eval_dataset"] = dataset["eval"]
        
        if self.data_format == "chat":
            def formatting_func(examples):
                texts = []
                for messages in examples["messages"]:
                    if self.config.data.system_message:
                        if not any(m.get("role") == "system" for m in messages):
                            messages = [
                                {"role": "system", "content": self.config.data.system_message}
                            ] + list(messages)
                    
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                return texts
            
            trainer_kwargs["formatting_func"] = formatting_func
        
        self.trainer = SFTTrainer(**trainer_kwargs)
        
        if self.config.tracking.wandb_enabled:
            self._setup_wandb()
        
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        self._save_model()
        
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training complete!")
        return metrics

    def _setup_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            wandb.init(
                project=self.config.tracking.wandb_project,
                entity=self.config.tracking.wandb_entity,
                name=self.config.tracking.wandb_run_name,
                tags=self.config.tracking.wandb_tags,
                config=self.config.to_dict(),
            )
            logger.info(f"WandB initialized: {self.config.tracking.wandb_project}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    def _save_model(self):
        """Save the trained model and tokenizer."""
        output_dir = Path(self.config.training.output_dir)
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.config.save(output_dir / "training_config.yaml")
        
        logger.info(f"Model saved to: {output_dir}")

    def evaluate(self, dataset: Optional[DatasetDict] = None) -> Dict[str, float]:
        """Evaluate the model on the eval dataset."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run train() first.")
        
        logger.info("Running evaluation...")
        
        eval_dataset = dataset["eval"] if dataset else None
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "float16",
    trust_remote_code: bool = True,
):
    """
    Merge LoRA adapter weights into base model.
    
    Args:
        base_model_path: Path or HF hub ID of base model.
        adapter_path: Path to saved adapter weights.
        output_dir: Directory to save merged model.
        torch_dtype: Torch dtype for merged model.
        trust_remote_code: Whether to trust remote code.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading base model: {base_model_path}")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    
    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapter weights...")
    merged_model = model.merge_and_unload()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_path)
    
    logger.info("Merge complete!")


def export_model(
    model_path: str,
    output_dir: str,
    export_format: str = "safetensors",
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
):
    """
    Export model to various formats.
    
    Args:
        model_path: Path to model to export.
        output_dir: Output directory.
        export_format: Format to export (safetensors, pytorch).
        push_to_hub: Whether to push to Hugging Face Hub.
        hub_model_id: Model ID for Hub upload.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if export_format == "safetensors":
        logger.info("Exporting to safetensors format...")
        model.save_pretrained(output_path, safe_serialization=True)
    elif export_format == "pytorch":
        logger.info("Exporting to PyTorch format...")
        model.save_pretrained(output_path, safe_serialization=False)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    tokenizer.save_pretrained(output_path)
    
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
    
    logger.info(f"Export complete: {output_path}")
