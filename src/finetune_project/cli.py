"""
Command-line interface for the fine-tuning pipeline.

Provides subcommands: prepare-data, train, evaluate, merge-adapter, export
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from finetune_project.config import Config, load_config, validate_config

app = typer.Typer(
    name="finetune",
    help="LLM Fine-Tuning Pipeline with LangChain - Train models on RunPod",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def print_config_summary(config: Config):
    """Print a summary of the configuration."""
    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", config.model.model_name_or_path)
    table.add_row("Output Dir", config.training.output_dir)
    table.add_row("Epochs", str(config.training.num_train_epochs))
    table.add_row("Max Steps", str(config.training.max_steps))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Max Seq Length", str(config.training.max_seq_length))
    table.add_row("PEFT Enabled", str(config.peft.enabled))
    table.add_row("Quantization", str(config.quantization.enabled))
    table.add_row("Data Path", config.data.input_path)
    table.add_row("LangChain Enabled", str(config.data.langchain.enabled))
    table.add_row("QA Generation", str(config.data.langchain.qa_generation_enabled))
    table.add_row("Augmentation", str(config.data.augmentation.enabled))
    
    console.print(table)


@app.command("prepare-data")
def prepare_data(
    config: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    input_path: Optional[str] = typer.Option(
        None,
        "--input", "-i",
        help="Override input data path",
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Override output dataset path",
    ),
    input_type: Optional[str] = typer.Option(
        None,
        "--type", "-t",
        help="Input type: auto, json, pdf, docx",
    ),
    enable_qa: bool = typer.Option(
        False,
        "--enable-qa",
        help="Enable QA generation from documents (requires API key)",
    ),
    enable_augmentation: bool = typer.Option(
        False,
        "--augment",
        help="Enable data augmentation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Prepare training data from JSON, PDF, or DOCX files.
    
    Uses LangChain for document loading and processing.
    Optionally generates Q&A pairs using Claude Haiku.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("📊 [bold]Preparing Training Data with LangChain[/bold]"))
    
    cfg = load_config(config)
    
    if input_path:
        cfg.data.input_path = input_path
    if output_path:
        cfg.data.output_dataset_path = output_path
    if input_type:
        cfg.data.input_type = input_type
    if enable_qa:
        cfg.data.langchain.qa_generation_enabled = True
    if enable_augmentation:
        cfg.data.augmentation.enabled = True
    
    issues = validate_config(cfg)
    for issue in issues:
        if issue.startswith("ERROR"):
            console.print(f"[red]{issue}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[yellow]{issue}[/yellow]")
    
    print_config_summary(cfg)
    
    from finetune_project.langchain_pipeline import LangChainDataPipeline
    
    pipeline = LangChainDataPipeline(cfg)
    dataset = pipeline.prepare()
    
    console.print(f"\n[green]✓ Data prepared successfully![/green]")
    console.print(f"  Train examples: {len(dataset['train'])}")
    if "eval" in dataset:
        console.print(f"  Eval examples: {len(dataset['eval'])}")
    console.print(f"  Output: {cfg.data.output_dataset_path}")


@app.command("train")
def train(
    config: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Override model name/path",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Override output directory",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Override number of epochs",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Train/fine-tune the model on prepared data.
    
    Runs the training loop with PEFT/LoRA if enabled.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("🚀 [bold]Starting Training[/bold]"))
    
    cfg = load_config(config)
    
    if model:
        cfg.model.model_name_or_path = model
    if output_dir:
        cfg.training.output_dir = output_dir
    if epochs:
        cfg.training.num_train_epochs = epochs
    if batch_size:
        cfg.training.per_device_train_batch_size = batch_size
    if learning_rate:
        cfg.training.learning_rate = learning_rate
    
    issues = validate_config(cfg)
    for issue in issues:
        if issue.startswith("ERROR"):
            console.print(f"[red]{issue}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[yellow]{issue}[/yellow]")
    
    print_config_summary(cfg)
    
    from finetune_project.langchain_pipeline import LangChainDataPipeline
    from finetune_project.trainer import FineTuneTrainer
    
    pipeline = LangChainDataPipeline(cfg)
    
    try:
        dataset = pipeline.load_prepared_dataset()
        console.print(f"[green]✓ Loaded prepared dataset[/green]")
    except FileNotFoundError:
        console.print("[yellow]Prepared dataset not found, running prepare-data...[/yellow]")
        dataset = pipeline.prepare()
    
    console.print(f"  Train examples: {len(dataset['train'])}")
    if "eval" in dataset:
        console.print(f"  Eval examples: {len(dataset['eval'])}")
    
    trainer = FineTuneTrainer(cfg)
    metrics = trainer.train(dataset)
    
    console.print(f"\n[green]✓ Training complete![/green]")
    console.print(f"  Output: {cfg.training.output_dir}")
    
    metrics_table = Table(title="Training Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics_table.add_row(key, f"{value:.4f}")
        else:
            metrics_table.add_row(key, str(value))
    
    console.print(metrics_table)


@app.command("evaluate")
def evaluate(
    config: Path = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Path to trained model",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Evaluate a trained model on the eval dataset.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("📈 [bold]Evaluating Model[/bold]"))
    
    cfg = load_config(config)
    
    if model_path:
        cfg.model.model_name_or_path = model_path
    
    from finetune_project.langchain_pipeline import LangChainDataPipeline
    from finetune_project.trainer import FineTuneTrainer
    
    pipeline = LangChainDataPipeline(cfg)
    dataset = pipeline.load_prepared_dataset()
    
    if "eval" not in dataset:
        console.print("[red]No evaluation split found in dataset[/red]")
        raise typer.Exit(1)
    
    trainer = FineTuneTrainer(cfg)
    trainer._load_model_and_tokenizer()
    metrics = trainer.evaluate(dataset)
    
    console.print(f"\n[green]✓ Evaluation complete![/green]")
    
    metrics_table = Table(title="Evaluation Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics_table.add_row(key, f"{value:.4f}")
        else:
            metrics_table.add_row(key, str(value))
    
    console.print(metrics_table)


@app.command("merge-adapter")
def merge_adapter_cmd(
    base_model: str = typer.Option(
        ...,
        "--base-model", "-b",
        help="Path or HF ID of base model",
    ),
    adapter_path: str = typer.Option(
        ...,
        "--adapter", "-a",
        help="Path to LoRA adapter weights",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory for merged model",
    ),
    dtype: str = typer.Option(
        "float16",
        "--dtype", "-d",
        help="Torch dtype: float16, bfloat16, float32",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Merge LoRA adapter weights into the base model.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("🔗 [bold]Merging Adapter[/bold]"))
    
    console.print(f"Base model: {base_model}")
    console.print(f"Adapter: {adapter_path}")
    console.print(f"Output: {output_dir}")
    
    from finetune_project.trainer import merge_adapter
    
    merge_adapter(
        base_model_path=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        torch_dtype=dtype,
    )
    
    console.print(f"\n[green]✓ Merge complete![/green]")
    console.print(f"  Output: {output_dir}")


@app.command("export")
def export_cmd(
    model_path: str = typer.Option(
        ...,
        "--model", "-m",
        help="Path to model to export",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory",
    ),
    format: str = typer.Option(
        "safetensors",
        "--format", "-f",
        help="Export format: safetensors, pytorch",
    ),
    push_to_hub: bool = typer.Option(
        False,
        "--push",
        help="Push to Hugging Face Hub",
    ),
    hub_model_id: Optional[str] = typer.Option(
        None,
        "--hub-id",
        help="Model ID for Hub upload",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Export model to various formats.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("📦 [bold]Exporting Model[/bold]"))
    
    from finetune_project.trainer import export_model
    
    export_model(
        model_path=model_path,
        output_dir=output_dir,
        export_format=format,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )
    
    console.print(f"\n[green]✓ Export complete![/green]")
    console.print(f"  Output: {output_dir}")


@app.command("validate")
def validate(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
):
    """
    Validate a configuration file.
    """
    console.print(Panel.fit("✅ [bold]Validating Configuration[/bold]"))
    
    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)
    
    issues = validate_config(cfg)
    
    if not issues:
        console.print("[green]✓ Configuration is valid![/green]")
        print_config_summary(cfg)
    else:
        has_errors = False
        for issue in issues:
            if issue.startswith("ERROR"):
                console.print(f"[red]{issue}[/red]")
                has_errors = True
            else:
                console.print(f"[yellow]{issue}[/yellow]")
        
        if has_errors:
            raise typer.Exit(1)
        else:
            console.print("\n[green]✓ Configuration is valid (with warnings)[/green]")
            print_config_summary(cfg)


@app.command("init")
def init_config(
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output", "-o",
        help="Output path for config file",
    ),
    template: str = typer.Option(
        "default",
        "--template", "-t",
        help="Template: default, json_sft, pdf_causal, qa_generation",
    ),
):
    """
    Generate a configuration file template.
    """
    console.print(Panel.fit("📝 [bold]Creating Config Template[/bold]"))
    
    cfg = Config()
    
    if template == "json_sft":
        cfg.data.input_type = "json"
        cfg.data.json.schema = "alpaca"
        cfg.peft.enabled = True
    elif template == "pdf_causal":
        cfg.data.input_type = "pdf"
        cfg.peft.enabled = True
        cfg.training.packing = True
    elif template == "qa_generation":
        cfg.data.input_type = "pdf"
        cfg.data.langchain.qa_generation_enabled = True
        cfg.data.langchain.qa_llm_provider = "anthropic"
        cfg.data.langchain.qa_model_name = "claude-3-haiku-20240307"
        cfg.peft.enabled = True
    
    cfg.save(output)
    console.print(f"[green]✓ Created config: {output}[/green]")


@app.command("generate-qa")
def generate_qa(
    input_path: str = typer.Option(
        ...,
        "--input", "-i",
        help="Path to PDF or document file/directory",
    ),
    output_path: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output JSONL file path",
    ),
    num_pairs: int = typer.Option(
        3,
        "--num-pairs", "-n",
        help="Number of Q&A pairs per chunk",
    ),
    chunk_size: int = typer.Option(
        1024,
        "--chunk-size",
        help="Text chunk size",
    ),
    provider: str = typer.Option(
        "anthropic",
        "--provider", "-p",
        help="LLM provider: anthropic, openai",
    ),
    model: str = typer.Option(
        "claude-3-haiku-20240307",
        "--model", "-m",
        help="LLM model name",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """
    Generate Q&A pairs from documents using LLM.
    
    Standalone command for QA generation without full training setup.
    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit("❓ [bold]Generating Q&A Pairs[/bold]"))
    
    import os
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    
    cfg = Config()
    cfg.data.input_path = input_path
    cfg.data.input_type = "pdf"
    cfg.data.pdf.chunk_size = chunk_size
    cfg.data.langchain.qa_generation_enabled = True
    cfg.data.langchain.qa_llm_provider = provider
    cfg.data.langchain.qa_model_name = model
    cfg.data.langchain.qa_pairs_per_chunk = num_pairs
    cfg.data.output_dataset_path = str(Path(output_path).parent)
    
    from finetune_project.langchain_pipeline import LangChainDataPipeline
    
    pipeline = LangChainDataPipeline(cfg)
    dataset = pipeline.prepare()
    
    # Save as JSONL
    import json
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset['train']:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]✓ Generated {len(dataset['train'])} examples[/green]")
    console.print(f"  Output: {output_file}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
