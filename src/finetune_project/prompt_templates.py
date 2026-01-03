"""
Prompt template management using LangChain.

Provides standardized prompt templates for:
- Instruction formatting (Alpaca, ChatML, custom)
- QA generation from documents
- Data augmentation templates
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Default Prompt Templates
# =============================================================================

# Alpaca-style instruction template
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_NO_INPUT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

# ChatML template
CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""

# QA Generation template for extracting Q&A pairs from text
QA_GENERATION_SYSTEM = """You are an expert at creating high-quality question-answer pairs from text.
Your task is to generate {num_pairs} diverse, informative question-answer pairs based on the given text.

Guidelines:
- Questions should be specific and answerable from the text
- Answers should be accurate and based only on the provided text
- Include a mix of factual, conceptual, and application questions
- Ensure questions cover different aspects of the text
- Keep answers concise but complete

Output format: Return a JSON array of objects with "question" and "answer" fields."""

QA_GENERATION_HUMAN = """Generate {num_pairs} question-answer pairs from the following text:

---
{text}
---

Return ONLY a valid JSON array, no other text."""

# Instruction variation templates (for augmentation)
INSTRUCTION_VARIATIONS = [
    "Based on the document, {original_instruction}",
    "According to the text, {original_instruction}",
    "From the information provided, {original_instruction}",
    "Using the context given, {original_instruction}",
    "Considering the content above, {original_instruction}",
    "With reference to the passage, {original_instruction}",
    "Drawing from the text, {original_instruction}",
    "As stated in the document, {original_instruction}",
]

# Paraphrase templates for simple augmentation
PARAPHRASE_PREFIXES = [
    "",
    "Please ",
    "Could you ",
    "Can you ",
    "I need you to ",
    "Help me ",
]

PARAPHRASE_SUFFIXES = [
    "",
    " Please be thorough.",
    " Be concise.",
    " Provide details.",
    " Explain clearly.",
]


class PromptTemplateManager:
    """
    Manages prompt templates using LangChain's template system.
    
    Supports:
    - Multiple instruction formats (Alpaca, ChatML, custom)
    - QA generation prompts
    - Augmentation templates
    """

    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        default_system_message: Optional[str] = None,
    ):
        """
        Initialize prompt template manager.
        
        Args:
            template_dir: Directory containing custom YAML templates.
            default_system_message: Default system message for chat formats.
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.default_system_message = default_system_message or "You are a helpful assistant."
        self._templates: Dict[str, Any] = {}
        self._load_default_templates()
        
        if self.template_dir and self.template_dir.exists():
            self._load_custom_templates()

    def _load_default_templates(self):
        """Load built-in default templates."""
        # Alpaca templates
        self._templates["alpaca"] = PromptTemplate(
            input_variables=["instruction", "input", "output"],
            template=ALPACA_TEMPLATE,
        )
        self._templates["alpaca_no_input"] = PromptTemplate(
            input_variables=["instruction", "output"],
            template=ALPACA_NO_INPUT_TEMPLATE,
        )
        
        # ChatML template
        self._templates["chatml"] = PromptTemplate(
            input_variables=["system", "user", "assistant"],
            template=CHATML_TEMPLATE,
        )
        
        # QA Generation template
        self._templates["qa_generation"] = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(QA_GENERATION_SYSTEM),
            HumanMessagePromptTemplate.from_template(QA_GENERATION_HUMAN),
        ])
        
        logger.debug("Loaded default prompt templates")

    def _load_custom_templates(self):
        """Load custom templates from template directory."""
        if not self.template_dir:
            return
            
        import yaml
        
        for yaml_file in self.template_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                if template_data and 'template' in template_data:
                    name = yaml_file.stem
                    self._templates[name] = PromptTemplate(
                        input_variables=template_data.get('input_variables', []),
                        template=template_data['template'],
                    )
                    logger.info(f"Loaded custom template: {name}")
            except Exception as e:
                logger.warning(f"Failed to load template {yaml_file}: {e}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def format_alpaca(
        self,
        instruction: str,
        input_text: str = "",
        output: str = "",
    ) -> str:
        """
        Format example in Alpaca style.
        
        Args:
            instruction: The instruction/task description.
            input_text: Optional input context.
            output: The expected output/response.
            
        Returns:
            Formatted prompt string.
        """
        if input_text:
            return self._templates["alpaca"].format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        else:
            return self._templates["alpaca_no_input"].format(
                instruction=instruction,
                output=output,
            )

    def format_chatml(
        self,
        user: str,
        assistant: str,
        system: Optional[str] = None,
    ) -> str:
        """
        Format example in ChatML style.
        
        Args:
            user: User message content.
            assistant: Assistant response.
            system: Optional system message.
            
        Returns:
            Formatted ChatML string.
        """
        return self._templates["chatml"].format(
            system=system or self.default_system_message,
            user=user,
            assistant=assistant,
        )

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Format messages list with optional system message injection.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_message: Optional system message to prepend.
            
        Returns:
            Formatted messages list.
        """
        result = []
        
        # Add system message if provided and not already present
        if system_message:
            if not messages or messages[0].get("role") != "system":
                result.append({"role": "system", "content": system_message})
        
        result.extend(messages)
        return result

    def get_qa_generation_prompt(
        self,
        text: str,
        num_pairs: int = 3,
    ) -> ChatPromptTemplate:
        """
        Get QA generation prompt for a text chunk.
        
        Args:
            text: Source text to generate Q&A from.
            num_pairs: Number of Q&A pairs to generate.
            
        Returns:
            Formatted ChatPromptTemplate ready for LLM invocation.
        """
        return self._templates["qa_generation"].format_messages(
            num_pairs=num_pairs,
            text=text,
        )

    def get_instruction_variations(
        self,
        original_instruction: str,
        num_variations: int = 2,
    ) -> List[str]:
        """
        Generate instruction variations for augmentation.
        
        Args:
            original_instruction: Original instruction text.
            num_variations: Number of variations to generate.
            
        Returns:
            List of instruction variations.
        """
        import random
        
        variations = [original_instruction]  # Always include original
        
        # Select random variation templates
        available_templates = INSTRUCTION_VARIATIONS.copy()
        random.shuffle(available_templates)
        
        for template in available_templates[:num_variations]:
            variation = template.format(original_instruction=original_instruction.lower())
            # Capitalize first letter
            variation = variation[0].upper() + variation[1:]
            variations.append(variation)
        
        return variations

    def get_paraphrase_variations(
        self,
        text: str,
        num_variations: int = 2,
    ) -> List[str]:
        """
        Generate simple paraphrase variations using templates.
        
        Args:
            text: Original text to paraphrase.
            num_variations: Number of variations to generate.
            
        Returns:
            List of paraphrased texts.
        """
        import random
        
        variations = [text]  # Always include original
        
        for _ in range(num_variations):
            prefix = random.choice(PARAPHRASE_PREFIXES)
            suffix = random.choice(PARAPHRASE_SUFFIXES)
            
            # Apply prefix (adjust case if needed)
            if prefix:
                modified = prefix + text[0].lower() + text[1:]
            else:
                modified = text
            
            # Apply suffix
            if suffix and not modified.endswith('.'):
                modified = modified.rstrip('.') + '.' + suffix
            elif suffix:
                modified = modified + suffix
            
            if modified not in variations:
                variations.append(modified)
        
        return variations

    def create_custom_template(
        self,
        name: str,
        template_string: str,
        input_variables: List[str],
    ):
        """
        Create and register a custom template.
        
        Args:
            name: Template name for retrieval.
            template_string: The template string with {variable} placeholders.
            input_variables: List of variable names used in template.
        """
        self._templates[name] = PromptTemplate(
            input_variables=input_variables,
            template=template_string,
        )
        logger.info(f"Created custom template: {name}")

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())


def create_chat_prompt_from_example(
    example: Dict[str, Any],
    schema: str = "auto",
    system_message: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert a training example to chat message format.
    
    Args:
        example: Training example dict.
        schema: Schema type (auto, alpaca, chatml, etc.)
        system_message: Optional system message.
        
    Returns:
        List of message dicts.
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Handle different schemas
    if schema == "chatml" or "messages" in example:
        # Already in messages format
        msgs = example.get("messages", [])
        messages.extend(msgs)
    
    elif schema == "alpaca" or "instruction" in example:
        # Alpaca format
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})
    
    elif "prompt" in example and "completion" in example:
        # Prompt/completion format
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["completion"]})
    
    else:
        # Try to infer format
        if "question" in example and "answer" in example:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
    
    return messages
