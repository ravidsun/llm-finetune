"""
Configuration management for the fine-tuning pipeline.

Supports YAML config files with environment variable overrides.
Includes LangChain-specific configuration options.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32
    attn_implementation: Optional[str] = None  # flash_attention_2, sdpa, eager


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "./output"
    seed: int = 42
    max_steps: int = -1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    optim: str = "paged_adamw_8bit"
    max_seq_length: int = 2048
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    packing: bool = True
    report_to: str = "none"  # none, wandb, tensorboard
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class PeftConfig:
    """Parameter-efficient fine-tuning configuration."""
    enabled: bool = True
    method: str = "lora"  # lora, qlora
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    modules_to_save: Optional[List[str]] = None


@dataclass
class QuantizationConfig:
    """Quantization configuration for memory efficiency."""
    enabled: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class JsonDataConfig:
    """JSON data configuration."""
    schema: str = "auto"  # auto, alpaca, chatml, openai_messages, custom
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    messages_field: str = "messages"
    system_field: str = "system"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    text_field: str = "text"


@dataclass
class PdfDataConfig:
    """PDF extraction configuration using LangChain."""
    extraction_backend: str = "pymupdf"  # pymupdf, pdfplumber, unstructured
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    strip_headers_footers: bool = True
    clean_whitespace: bool = True
    # LangChain text splitter options
    splitter_type: str = "recursive"  # recursive, character, token
    separators: Optional[List[str]] = None  # Custom separators for recursive splitter


@dataclass
class LangChainConfig:
    """LangChain-specific configuration."""
    enabled: bool = True
    # Document loader settings
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)
    # QA Generation settings (optional, requires LLM API)
    qa_generation_enabled: bool = False  # OFF by default
    qa_llm_provider: str = "anthropic"  # anthropic, openai
    qa_model_name: str = "claude-3-haiku-20240307"  # Cost-effective default
    qa_pairs_per_chunk: int = 3
    qa_temperature: float = 0.3
    qa_max_tokens: int = 1024
    # Prompt template settings
    prompt_template_dir: Optional[str] = None  # Custom templates directory
    default_system_message: Optional[str] = None


@dataclass
class AugmentationConfig:
    """Data augmentation configuration (deterministic, no API calls)."""
    enabled: bool = False
    # Instruction variations
    instruction_variations: bool = True
    num_instruction_variations: int = 2
    # Text transformations
    case_variations: bool = False  # Add lowercase/uppercase variants
    whitespace_normalization: bool = True
    # Paraphrase templates (rule-based)
    use_paraphrase_templates: bool = True
    # Shuffle order for multi-turn conversations
    shuffle_conversation_order: bool = False


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    input_path: str = "/workspace/data"
    input_type: str = "auto"  # auto, json, pdf, docx
    output_dataset_path: str = "/workspace/processed_data"
    train_split: float = 0.9
    eval_split: float = 0.1
    shuffle: bool = True
    # Sub-configs
    json: JsonDataConfig = field(default_factory=JsonDataConfig)
    pdf: PdfDataConfig = field(default_factory=PdfDataConfig)
    langchain: LangChainConfig = field(default_factory=LangChainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    # Chat template settings
    chat_template: Optional[str] = None
    system_message: Optional[str] = None


@dataclass
class TrackingConfig:
    """Experiment tracking configuration."""
    wandb_enabled: bool = False
    wandb_project: str = "llm-finetune"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)


@dataclass
class RunpodConfig:
    """RunPod-specific configuration."""
    volume_path: str = "/workspace"
    hf_cache_dir: str = "/workspace/hf_cache"
    data_path: str = "/workspace/data"
    output_path: str = "/workspace/output"


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runpod: RunpodConfig = field(default_factory=RunpodConfig)

    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Override config values with environment variables."""
        # Model overrides
        if os.getenv("MODEL_NAME"):
            self.model.model_name_or_path = os.getenv("MODEL_NAME")
        
        # Training overrides
        if os.getenv("OUTPUT_DIR"):
            self.training.output_dir = os.getenv("OUTPUT_DIR")
        if os.getenv("MAX_STEPS"):
            self.training.max_steps = int(os.getenv("MAX_STEPS"))
        if os.getenv("NUM_EPOCHS"):
            self.training.num_train_epochs = int(os.getenv("NUM_EPOCHS"))
        if os.getenv("BATCH_SIZE"):
            self.training.per_device_train_batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("LEARNING_RATE"):
            self.training.learning_rate = float(os.getenv("LEARNING_RATE"))
        if os.getenv("MAX_SEQ_LENGTH"):
            self.training.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH"))

        # Data overrides
        if os.getenv("DATA_PATH"):
            self.data.input_path = os.getenv("DATA_PATH")
        if os.getenv("INPUT_TYPE"):
            self.data.input_type = os.getenv("INPUT_TYPE")

        # LangChain QA generation override
        if os.getenv("ENABLE_QA_GENERATION"):
            self.data.langchain.qa_generation_enabled = os.getenv("ENABLE_QA_GENERATION").lower() == "true"
        if os.getenv("ANTHROPIC_API_KEY"):
            # QA generation available
            pass
        if os.getenv("QA_LLM_PROVIDER"):
            self.data.langchain.qa_llm_provider = os.getenv("QA_LLM_PROVIDER")

        # Tracking overrides
        if os.getenv("WANDB_PROJECT"):
            self.tracking.wandb_project = os.getenv("WANDB_PROJECT")
            self.tracking.wandb_enabled = True
        if os.getenv("WANDB_ENTITY"):
            self.tracking.wandb_entity = os.getenv("WANDB_ENTITY")

        # RunPod overrides
        if os.getenv("RUNPOD_VOLUME_PATH"):
            self.runpod.volume_path = os.getenv("RUNPOD_VOLUME_PATH")
        if os.getenv("HF_HOME"):
            self.runpod.hf_cache_dir = os.getenv("HF_HOME")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        def _asdict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _asdict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_asdict(v) for v in obj]
            return obj
        return _asdict(self)

    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {path}")


def _build_nested_config(config_dict: Dict[str, Any], key: str, config_class):
    """Build nested config from dictionary, handling missing keys."""
    nested_dict = config_dict.get(key, {})
    if nested_dict is None:
        nested_dict = {}
    
    # Get valid fields for the config class
    valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in nested_dict.items() if k in valid_fields}
    
    return config_class(**filtered_dict)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Path to YAML config file. Uses defaults if None.
        
    Returns:
        Config object with all settings.
    """
    # Load .env file if present
    load_dotenv()
    
    if config_path is None:
        logger.info("No config file provided, using defaults with env overrides")
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Build nested data config
    data_dict = config_dict.get('data', {}) or {}
    data_config = DataConfig(
        **{k: v for k, v in data_dict.items() 
           if k not in ['json', 'pdf', 'langchain', 'augmentation'] and k in DataConfig.__dataclass_fields__},
        json=_build_nested_config(data_dict, 'json', JsonDataConfig),
        pdf=_build_nested_config(data_dict, 'pdf', PdfDataConfig),
        langchain=_build_nested_config(data_dict, 'langchain', LangChainConfig),
        augmentation=_build_nested_config(data_dict, 'augmentation', AugmentationConfig),
    )
    
    # Build main config
    config = Config(
        model=_build_nested_config(config_dict, 'model', ModelConfig),
        training=_build_nested_config(config_dict, 'training', TrainingConfig),
        peft=_build_nested_config(config_dict, 'peft', PeftConfig),
        quantization=_build_nested_config(config_dict, 'quantization', QuantizationConfig),
        data=data_config,
        tracking=_build_nested_config(config_dict, 'tracking', TrackingConfig),
        runpod=_build_nested_config(config_dict, 'runpod', RunpodConfig),
    )
    
    return config


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate.
        
    Returns:
        List of warning/error messages.
    """
    issues = []
    
    # Check model path
    if not config.model.model_name_or_path:
        issues.append("ERROR: model.model_name_or_path is required")
    
    # Check data path
    data_path = Path(config.data.input_path)
    if not data_path.exists():
        issues.append(f"WARNING: Data path does not exist: {data_path}")
    
    # Check output directory
    output_dir = Path(config.training.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        issues.append(f"WARNING: Output directory is not empty: {output_dir}")
    
    # Check PEFT configuration
    if config.peft.enabled and config.peft.method not in ['lora', 'qlora']:
        issues.append(f"ERROR: Unsupported PEFT method: {config.peft.method}")
    
    # Check quantization compatibility
    if config.quantization.load_in_4bit and config.quantization.load_in_8bit:
        issues.append("ERROR: Cannot enable both 4-bit and 8-bit quantization")
    
    # Check training parameters
    if config.training.max_steps <= 0 and config.training.num_train_epochs <= 0:
        issues.append("ERROR: Either max_steps or num_train_epochs must be positive")
    
    if config.training.per_device_train_batch_size < 1:
        issues.append("ERROR: Batch size must be at least 1")
    
    if config.training.learning_rate <= 0:
        issues.append("ERROR: Learning rate must be positive")
    
    # Check split ratios
    if config.data.train_split + config.data.eval_split > 1.0:
        issues.append("ERROR: train_split + eval_split cannot exceed 1.0")
    
    # Check QA generation requirements
    if config.data.langchain.qa_generation_enabled:
        provider = config.data.langchain.qa_llm_provider
        if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("WARNING: QA generation enabled but ANTHROPIC_API_KEY not set")
        elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            issues.append("WARNING: QA generation enabled but OPENAI_API_KEY not set")
    
    return issues
