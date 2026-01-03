"""
LLM Fine-Tuning Project with LangChain Integration

A production-ready pipeline for fine-tuning open-source LLMs on RunPod.
Features:
- LangChain-powered document processing and prompt templating
- PEFT/LoRA support with quantization
- Multiple data formats (JSON, PDF, DOCX)
- Optional LLM-based QA generation
- Flexible configuration via YAML
"""

__version__ = "2.0.0"
__author__ = "ML Engineer"

from finetune_project.config import Config, load_config
from finetune_project.langchain_pipeline import LangChainDataPipeline
from finetune_project.trainer import FineTuneTrainer
from finetune_project.augmentation import DataAugmenter
from finetune_project.prompt_templates import PromptTemplateManager

__all__ = [
    "Config",
    "load_config",
    "LangChainDataPipeline",
    "FineTuneTrainer",
    "DataAugmenter",
    "PromptTemplateManager",
]
