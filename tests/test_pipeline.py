"""
Smoke tests for the fine-tuning pipeline with LangChain.

Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config_loads(self):
        """Test that default config can be created."""
        from finetune_project.config import Config
        
        config = Config()
        assert config.model.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert config.peft.enabled is True
        assert config.quantization.load_in_4bit is True
        assert config.data.langchain.enabled is True
        assert config.data.langchain.qa_generation_enabled is False  # OFF by default

    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from finetune_project.config import Config, load_config
        
        config = Config()
        config_path = tmp_path / "test_config.yaml"
        config.save(config_path)
        
        loaded = load_config(config_path)
        assert loaded.model.model_name_or_path == config.model.model_name_or_path

    def test_config_validation(self):
        """Test config validation catches errors."""
        from finetune_project.config import Config, validate_config
        
        config = Config()
        config.training.max_steps = 0
        config.training.num_train_epochs = 0
        
        issues = validate_config(config)
        assert any("max_steps" in issue or "epochs" in issue for issue in issues)

    def test_langchain_config_defaults(self):
        """Test LangChain config has correct defaults."""
        from finetune_project.config import Config
        
        config = Config()
        assert config.data.langchain.qa_generation_enabled is False
        assert config.data.langchain.qa_llm_provider == "anthropic"
        assert "haiku" in config.data.langchain.qa_model_name.lower()


class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_prompt_manager_initialization(self):
        """Test prompt manager can be initialized."""
        from finetune_project.prompt_templates import PromptTemplateManager
        
        manager = PromptTemplateManager()
        assert "alpaca" in manager.list_templates()
        assert "chatml" in manager.list_templates()
        assert "qa_generation" in manager.list_templates()

    def test_alpaca_formatting(self):
        """Test Alpaca format generation."""
        from finetune_project.prompt_templates import PromptTemplateManager
        
        manager = PromptTemplateManager()
        result = manager.format_alpaca(
            instruction="What is 2+2?",
            input_text="",
            output="4"
        )
        
        assert "### Instruction:" in result
        assert "What is 2+2?" in result
        assert "### Response:" in result
        assert "4" in result

    def test_instruction_variations(self):
        """Test instruction variation generation."""
        from finetune_project.prompt_templates import PromptTemplateManager
        
        manager = PromptTemplateManager()
        variations = manager.get_instruction_variations(
            "Explain quantum computing",
            num_variations=3
        )
        
        assert len(variations) >= 2
        assert "Explain quantum computing" in variations  # Original included


class TestAugmentation:
    """Test data augmentation functionality."""

    def test_augmenter_initialization(self):
        """Test augmenter can be initialized."""
        from finetune_project.config import AugmentationConfig
        from finetune_project.augmentation import DataAugmenter
        
        config = AugmentationConfig(enabled=True)
        augmenter = DataAugmenter(config, seed=42)
        assert augmenter is not None

    def test_augmentation_disabled(self):
        """Test augmentation returns original when disabled."""
        from finetune_project.config import AugmentationConfig
        from finetune_project.augmentation import DataAugmenter
        
        config = AugmentationConfig(enabled=False)
        augmenter = DataAugmenter(config)
        
        examples = [{"instruction": "Test", "output": "Response"}]
        result = augmenter.augment_dataset(examples)
        
        assert len(result) == len(examples)

    def test_alpaca_augmentation(self):
        """Test Alpaca format augmentation."""
        from finetune_project.config import AugmentationConfig
        from finetune_project.augmentation import DataAugmenter
        
        config = AugmentationConfig(
            enabled=True,
            instruction_variations=True,
            num_instruction_variations=2
        )
        augmenter = DataAugmenter(config, seed=42)
        
        examples = [{"instruction": "Explain machine learning", "input": "", "output": "ML is..."}]
        result = augmenter.augment_dataset(examples)
        
        assert len(result) >= 2  # Original + at least 1 variation


class TestLangChainPipeline:
    """Test LangChain data pipeline."""

    def test_json_schema_detection(self, tmp_path):
        """Test JSON schema auto-detection."""
        from finetune_project.config import Config
        from finetune_project.langchain_pipeline import LangChainDataPipeline
        
        # Create alpaca-style JSON
        data = [
            {"instruction": "What is 2+2?", "input": "", "output": "4"},
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
        ]
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        config = Config()
        config.data.input_path = str(json_file)
        config.data.output_dataset_path = str(tmp_path / "output")
        
        pipeline = LangChainDataPipeline(config)
        schema = pipeline._detect_json_schema(data)
        
        assert schema == "alpaca"

    def test_jsonl_loading(self, tmp_path):
        """Test JSONL file loading."""
        from finetune_project.config import Config
        from finetune_project.langchain_pipeline import LangChainDataPipeline
        
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"instruction": "Test 1", "input": "", "output": "Response 1"}\n')
            f.write('{"instruction": "Test 2", "input": "", "output": "Response 2"}\n')
        
        config = Config()
        config.data.input_path = str(jsonl_file)
        
        pipeline = LangChainDataPipeline(config)
        examples = pipeline._load_json_files(jsonl_file)
        
        assert len(examples) == 2
        assert examples[0]["instruction"] == "Test 1"

    def test_text_splitter_initialization(self, tmp_path):
        """Test LangChain text splitter is initialized."""
        from finetune_project.config import Config
        from finetune_project.langchain_pipeline import LangChainDataPipeline
        
        config = Config()
        config.data.pdf.chunk_size = 500
        config.data.pdf.splitter_type = "recursive"
        
        pipeline = LangChainDataPipeline(config)
        
        assert pipeline.text_splitter is not None
        # Test splitting works
        from langchain_core.documents import Document
        doc = Document(page_content="This is a test. " * 100)
        chunks = pipeline.text_splitter.split_documents([doc])
        assert len(chunks) >= 1


class TestCLI:
    """Test CLI functionality."""

    def test_cli_help(self):
        """Test CLI help command works."""
        from typer.testing import CliRunner
        from finetune_project.cli import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "prepare-data" in result.stdout
        assert "train" in result.stdout
        assert "generate-qa" in result.stdout

    def test_init_command(self, tmp_path):
        """Test config initialization command."""
        from typer.testing import CliRunner
        from finetune_project.cli import app
        
        runner = CliRunner()
        output_path = tmp_path / "test_config.yaml"
        
        result = runner.invoke(app, ["init", "--output", str(output_path)])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_init_qa_template(self, tmp_path):
        """Test QA generation template initialization."""
        from typer.testing import CliRunner
        from finetune_project.cli import app
        
        runner = CliRunner()
        output_path = tmp_path / "qa_config.yaml"
        
        result = runner.invoke(app, [
            "init", 
            "--output", str(output_path),
            "--template", "qa_generation"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify QA generation is enabled in the template
        import yaml
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config['data']['langchain']['qa_generation_enabled'] is True


class TestImports:
    """Test that all modules can be imported."""

    def test_import_config(self):
        from finetune_project.config import Config, load_config

    def test_import_langchain_pipeline(self):
        from finetune_project.langchain_pipeline import LangChainDataPipeline

    def test_import_trainer(self):
        from finetune_project.trainer import FineTuneTrainer, merge_adapter

    def test_import_cli(self):
        from finetune_project.cli import app

    def test_import_prompt_templates(self):
        from finetune_project.prompt_templates import PromptTemplateManager

    def test_import_augmentation(self):
        from finetune_project.augmentation import DataAugmenter


class TestLangChainImports:
    """Test LangChain imports work correctly."""

    def test_langchain_core_imports(self):
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate

    def test_langchain_text_splitters(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=100)
        result = splitter.split_text("This is a test. " * 50)
        assert len(result) >= 1
