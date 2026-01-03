# Visual Diagrams

This document contains Mermaid diagrams that render automatically on GitHub.

## System Architecture Overview

```mermaid
graph TB
    subgraph Input["📥 INPUT LAYER"]
        JSON[JSON Files]
        PDF[PDF Documents]
        DOCX[DOCX Files]
        TXT[Text Files]
    end

    subgraph Processing["⚙️ DATA PROCESSING LAYER"]
        subgraph LangChain["LangChain Pipeline"]
            Loaders[Document Loaders<br/>PDFLoader, DocxLoader, TextLoader]
            Splitters[Text Splitters<br/>Recursive, Character, Token]
        end

        Direct[Direct Processing<br/>Format as training data]
        QAGen[QA Generation<br/>Claude Haiku API]

        Templates[Prompt Templates<br/>Alpaca, ChatML, Custom]

        subgraph Augmentation["Data Augmentation"]
            InstVar[Instruction Variations]
            Para[Paraphrase Templates]
            Norm[Whitespace Normalization]
        end
    end

    subgraph Training["🎓 TRAINING LAYER"]
        BaseModel[Base Model<br/>Llama/Qwen/Mistral]
        LoRA[PEFT/LoRA Config<br/>Rank, Alpha, Target Modules]
        Trainer[TRL SFTTrainer<br/>Sequence Packing, Chat Templates]
    end

    subgraph Output["📤 OUTPUT LAYER"]
        Adapter[LoRA Adapter Weights]
        Merged[Merged Model]
        Export[Exported Model<br/>GGUF/Quantized]
    end

    JSON --> Loaders
    PDF --> Loaders
    DOCX --> Loaders
    TXT --> Loaders

    Loaders --> Splitters
    Splitters --> Direct
    Splitters --> QAGen

    Direct --> Templates
    QAGen --> Templates

    Templates --> InstVar
    InstVar --> Para
    Para --> Norm

    Norm --> BaseModel
    BaseModel --> LoRA
    LoRA --> Trainer

    Trainer --> Adapter
    Adapter --> Merged
    Merged --> Export

    classDef inputClass fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef trainClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px

    class JSON,PDF,DOCX,TXT inputClass
    class Loaders,Splitters,Direct,QAGen,Templates,InstVar,Para,Norm processClass
    class BaseModel,LoRA,Trainer trainClass
    class Adapter,Merged,Export outputClass
```

## Data Preparation Workflow

```mermaid
flowchart TD
    Start([Start]) --> LoadConfig[Load config.yaml]
    LoadConfig --> CheckInput{Input Type?}

    CheckInput -->|JSON| DirectJSON[Load JSON/JSONL]
    CheckInput -->|PDF| LoadPDF[LangChain PDFLoader]
    CheckInput -->|DOCX/TXT| LoadDoc[LangChain DocxLoader/TextLoader]

    LoadPDF --> Split[Text Splitting<br/>Recursive/Character/Token]
    LoadDoc --> Split

    Split --> QACheck{Generate QA?}

    QACheck -->|Yes| QAGen[Generate Q&A Pairs<br/>Claude Haiku API]
    QACheck -->|No| Format[Apply Prompt Templates<br/>Alpaca/ChatML/Custom]

    QAGen --> Format
    DirectJSON --> Format

    Format --> AugCheck{Augmentation?}

    AugCheck -->|Yes| Augment[Data Augmentation<br/>Variations & Paraphrasing]
    AugCheck -->|No| Save[Save to JSONL/TXT]

    Augment --> Save
    Save --> End([End - Ready for Training])

    classDef startEnd fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    classDef decision fill:#ff9800,stroke:#e65100,stroke-width:2px
    classDef process fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff

    class Start,End startEnd
    class CheckInput,QACheck,AugCheck decision
    class LoadConfig,DirectJSON,LoadPDF,LoadDoc,Split,QAGen,Format,Augment,Save process
```

## Training Pipeline

```mermaid
flowchart TD
    Start([Start Training]) --> Config[Load Training Config]
    Config --> Model[Initialize Base Model<br/>Download from Hugging Face]

    Model --> Quant{QLoRA<br/>Enabled?}
    Quant -->|Yes| QLoRA[Apply 4-bit Quantization]
    Quant -->|No| LoRAConf[Configure LoRA Adapter]

    QLoRA --> LoRAConf
    LoRAConf --> Dataset[Load Prepared Dataset<br/>Tokenize & Pack Sequences]

    Dataset --> SFT[Initialize SFTTrainer<br/>Set Hyperparameters]

    SFT --> Train[Training Loop]

    Train --> Epoch{More<br/>Epochs?}
    Epoch -->|Yes| Forward[Forward Pass]
    Forward --> Loss[Compute Loss]
    Loss --> Backward[Backward Pass]
    Backward --> Update[Update Weights]
    Update --> Log[Log Metrics]
    Log --> Train

    Epoch -->|No| SaveAdapter[Save LoRA Adapter<br/>adapter_model.safetensors]

    SaveAdapter --> MergeQ{Merge<br/>Adapter?}
    MergeQ -->|Yes| Merge[Merge with Base Model]
    MergeQ -->|No| ExportQ{Export?}

    Merge --> ExportQ
    ExportQ -->|Yes| Export[Export to GGUF/Quantize]
    ExportQ -->|No| End([Training Complete])

    Export --> End

    classDef startEnd fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    classDef decision fill:#ff9800,stroke:#e65100,stroke-width:2px
    classDef process fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
    classDef train fill:#9c27b0,stroke:#4a148c,stroke-width:2px,color:#fff

    class Start,End startEnd
    class Quant,Epoch,MergeQ,ExportQ decision
    class Config,Model,QLoRA,LoRAConf,Dataset,SFT,SaveAdapter,Merge,Export process
    class Train,Forward,Loss,Backward,Update,Log train
```

## Component Interaction

```mermaid
graph TB
    subgraph CLI["CLI Layer (cli.py)"]
        Init[init command]
        Validate[validate command]
        Prepare[prepare-data command]
        Train[train command]
        Eval[evaluate command]
        GenQA[generate-qa command]
        MergeCmd[merge-adapter command]
        ExportCmd[export command]
    end

    subgraph Config["Configuration Layer"]
        ConfigPy[config.py<br/>FineTuneConfig]
        YAML[config.yaml]
    end

    subgraph Process["Processing Layer"]
        LCP[langchain_pipeline.py<br/>Document Processing]
        PT[prompt_templates.py<br/>Template Management]
        Aug[augmentation.py<br/>Data Augmentation]
    end

    subgraph Train["Training Layer"]
        Trainer[trainer.py<br/>Model Training & Export]
    end

    subgraph External["External Dependencies"]
        HF[Hugging Face<br/>Transformers, PEFT, TRL]
        LC[LangChain<br/>Document Loaders]
        Claude[Anthropic Claude<br/>QA Generation]
    end

    Init --> ConfigPy
    Validate --> ConfigPy
    Prepare --> ConfigPy
    Train --> ConfigPy

    ConfigPy --> YAML

    Prepare --> LCP
    Prepare --> PT
    Prepare --> Aug

    LCP --> LC
    LCP --> Claude
    PT --> Aug

    Train --> Trainer
    Trainer --> HF

    GenQA --> LCP
    MergeCmd --> Trainer
    ExportCmd --> Trainer
    Eval --> Trainer

    classDef cliClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef configClass fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef trainClass fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef extClass fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class Init,Validate,Prepare,Train,Eval,GenQA,MergeCmd,ExportCmd cliClass
    class ConfigPy,YAML configClass
    class LCP,PT,Aug processClass
    class Trainer trainClass
    class HF,LC,Claude extClass
```

## RunPod Deployment Flow

```mermaid
flowchart LR
    Dev[Developer] -->|1. Create Pod| RunPod[RunPod Platform]
    RunPod -->|2. Deploy| Pod[GPU Pod<br/>RTX 4090/A40/A100]

    Pod -->|3. SSH/Web Terminal| Access[Access Pod]
    Access -->|4. Clone Repo| Clone[git clone]
    Clone -->|5. Run Setup| Setup[bash scripts/setup.sh]

    Setup -->|6. Install| Deps[Install Dependencies<br/>LangChain, PEFT, TRL]
    Deps -->|7. Configure| Env[Set Environment Variables<br/>HF_TOKEN, ANTHROPIC_API_KEY]

    Env -->|8. Prepare| Data[Prepare Data<br/>python -m finetune_project prepare-data]
    Data -->|9. Start| Training[Start Training<br/>bash scripts/runpod_start.sh]

    Training -->|10. Monitor| Logs[View Logs & Metrics<br/>WandB/TensorBoard]
    Logs -->|11. Complete| Save[Save to /workspace<br/>LoRA Adapters]

    Save -->|12. Download| Local[Download to Local Machine]

    classDef devClass fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#fff
    classDef podClass fill:#2196f3,stroke:#0d47a1,stroke-width:2px,color:#fff
    classDef processClass fill:#ff9800,stroke:#e65100,stroke-width:2px

    class Dev,Local devClass
    class RunPod,Pod,Access podClass
    class Clone,Setup,Deps,Env,Data,Training,Logs,Save processClass
```

## Class Diagram

```mermaid
classDiagram
    class FineTuneConfig {
        +ModelConfig model
        +DataConfig data
        +TrainingConfig training
        +from_yaml() FineTuneConfig
        +to_yaml() str
        +validate() bool
    }

    class ModelConfig {
        +str model_name
        +int lora_rank
        +int lora_alpha
        +float lora_dropout
        +list target_modules
        +bool use_qlora
    }

    class DataConfig {
        +str input_path
        +str output_path
        +str input_type
        +LangChainConfig langchain
        +AugmentConfig augmentation
        +PDFConfig pdf
    }

    class TrainingConfig {
        +int num_epochs
        +int batch_size
        +float learning_rate
        +int max_seq_length
        +bool gradient_checkpointing
        +str output_dir
    }

    class LangChainConfig {
        +bool enabled
        +bool qa_generation_enabled
        +str qa_llm_provider
        +str qa_model_name
        +int qa_pairs_per_chunk
    }

    class AugmentConfig {
        +bool enabled
        +bool instruction_variations
        +int num_instruction_variations
        +bool use_paraphrase_templates
    }

    class PromptTemplateManager {
        +format_alpaca() str
        +format_chatml() str
        +format_custom() str
        +get_instruction_variations() list
    }

    class DataAugmenter {
        +AugmentConfig config
        +augment_instruction() list
        +paraphrase() str
        +normalize() str
        +augment_dataset() Dataset
    }

    class LangChainPipeline {
        +load_documents() list
        +split_text() list
        +generate_qa_pairs() list
        +process_pipeline() Dataset
    }

    FineTuneConfig --> ModelConfig
    FineTuneConfig --> DataConfig
    FineTuneConfig --> TrainingConfig
    DataConfig --> LangChainConfig
    DataConfig --> AugmentConfig

    LangChainPipeline --> LangChainConfig
    LangChainPipeline --> PromptTemplateManager
    DataAugmenter --> AugmentConfig
```

## Sequence Diagram: QA Generation Flow

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Config
    participant LCP as LangChain Pipeline
    participant Claude as Claude API
    participant PT as Prompt Templates
    participant File as Output File

    User->>CLI: prepare-data --enable-qa
    CLI->>Config: Load config.yaml
    Config-->>CLI: Configuration loaded

    CLI->>LCP: process_pipeline(config)
    LCP->>LCP: load_documents(PDF)
    LCP->>LCP: split_text(chunks)

    loop For each chunk
        LCP->>Claude: Generate QA pairs
        Claude-->>LCP: Q&A JSON response
        LCP->>PT: format_alpaca(qa_pair)
        PT-->>LCP: Formatted text
    end

    LCP->>File: Save to JSONL
    File-->>CLI: Processing complete
    CLI-->>User: Data ready for training
```

## State Diagram: Training Process

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> LoadingModel: Config validated
    LoadingModel --> ConfiguringLoRA: Model loaded
    ConfiguringLoRA --> LoadingDataset: LoRA configured
    LoadingDataset --> Training: Dataset ready

    Training --> Epoch
    Epoch --> ForwardPass
    ForwardPass --> ComputeLoss
    ComputeLoss --> BackwardPass
    BackwardPass --> UpdateWeights
    UpdateWeights --> LogMetrics
    LogMetrics --> CheckEpoch

    CheckEpoch --> Epoch: More epochs
    CheckEpoch --> SavingAdapter: Training complete

    SavingAdapter --> [*]: Success

    Training --> Error: Failed
    Error --> [*]
```
