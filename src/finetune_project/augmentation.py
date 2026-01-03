"""
Data augmentation module with deterministic transformations.

Provides simple, rule-based augmentation techniques that don't require API calls:
- Instruction variations using templates
- Text normalization
- Paraphrase templates
- Case variations
"""

import logging
import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from finetune_project.config import AugmentationConfig
from finetune_project.prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Applies deterministic data augmentation to training examples.
    
    All augmentation methods are rule-based and don't require external API calls.
    """

    def __init__(
        self,
        config: AugmentationConfig,
        seed: int = 42,
    ):
        """
        Initialize data augmenter.
        
        Args:
            config: Augmentation configuration.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.prompt_manager = PromptTemplateManager()
        
        # Instruction transformation patterns
        self._instruction_transforms = [
            (r'^(explain|describe)\s+', 'Provide an explanation of '),
            (r'^(what is|what are)\s+', 'Define '),
            (r'^(how to|how do you)\s+', 'Explain the process to '),
            (r'^(list|enumerate)\s+', 'Provide a list of '),
            (r'^(compare)\s+', 'Make a comparison of '),
            (r'^(summarize)\s+', 'Provide a summary of '),
        ]
        
        # Question word synonyms
        self._question_synonyms = {
            'what': ['which', 'what exactly'],
            'how': ['in what way', 'by what means'],
            'why': ['for what reason', 'what causes'],
            'when': ['at what time', 'during which period'],
            'where': ['in what place', 'at which location'],
        }

    def augment_dataset(
        self,
        examples: List[Dict[str, Any]],
        max_augmented_per_example: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Augment a list of training examples.
        
        Args:
            examples: List of training examples.
            max_augmented_per_example: Max augmented versions per original.
            
        Returns:
            List of original + augmented examples.
        """
        if not self.config.enabled:
            logger.info("Augmentation disabled, returning original examples")
            return examples
        
        augmented = []
        
        for example in examples:
            # Always include original
            augmented.append(example)
            
            # Generate augmented versions
            augmented_versions = self._augment_single(
                example,
                max_versions=max_augmented_per_example,
            )
            augmented.extend(augmented_versions)
        
        logger.info(
            f"Augmented {len(examples)} examples to {len(augmented)} "
            f"(+{len(augmented) - len(examples)} augmented)"
        )
        
        return augmented

    def _augment_single(
        self,
        example: Dict[str, Any],
        max_versions: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented versions of a single example.
        
        Args:
            example: Original training example.
            max_versions: Maximum augmented versions to generate.
            
        Returns:
            List of augmented examples (not including original).
        """
        augmented = []
        
        # Determine example type
        if "messages" in example:
            # Chat format
            augmented.extend(self._augment_messages(example, max_versions))
        elif "instruction" in example:
            # Alpaca format
            augmented.extend(self._augment_alpaca(example, max_versions))
        elif "text" in example:
            # Text-only format (causal LM)
            augmented.extend(self._augment_text(example, max_versions))
        elif "prompt" in example:
            # Prompt/completion format
            augmented.extend(self._augment_prompt_completion(example, max_versions))
        
        return augmented

    def _augment_alpaca(
        self,
        example: Dict[str, Any],
        max_versions: int,
    ) -> List[Dict[str, Any]]:
        """Augment Alpaca-style example."""
        augmented = []
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        # Instruction variations
        if self.config.instruction_variations and instruction:
            variations = self.prompt_manager.get_instruction_variations(
                instruction,
                num_variations=self.config.num_instruction_variations,
            )
            
            for var_instruction in variations[1:max_versions + 1]:  # Skip original
                new_example = deepcopy(example)
                new_example["instruction"] = var_instruction
                new_example["_augmented"] = True
                augmented.append(new_example)
        
        # Apply text transformations
        if self.config.use_paraphrase_templates and instruction:
            transformed = self._transform_instruction(instruction)
            if transformed != instruction:
                new_example = deepcopy(example)
                new_example["instruction"] = transformed
                new_example["_augmented"] = True
                if len(augmented) < max_versions:
                    augmented.append(new_example)
        
        return augmented[:max_versions]

    def _augment_messages(
        self,
        example: Dict[str, Any],
        max_versions: int,
    ) -> List[Dict[str, Any]]:
        """Augment chat messages format."""
        augmented = []
        messages = example.get("messages", [])
        
        if not messages:
            return augmented
        
        # Find user messages to vary
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i < len(messages) - 1:
                user_content = msg.get("content", "")
                
                if self.config.instruction_variations:
                    variations = self.prompt_manager.get_instruction_variations(
                        user_content,
                        num_variations=1,
                    )
                    
                    for var_content in variations[1:]:  # Skip original
                        new_example = deepcopy(example)
                        new_example["messages"][i]["content"] = var_content
                        new_example["_augmented"] = True
                        augmented.append(new_example)
                        
                        if len(augmented) >= max_versions:
                            break
                
                if len(augmented) >= max_versions:
                    break
        
        return augmented[:max_versions]

    def _augment_text(
        self,
        example: Dict[str, Any],
        max_versions: int,
    ) -> List[Dict[str, Any]]:
        """Augment text-only example for causal LM."""
        augmented = []
        text = example.get("text", "")
        
        if not text:
            return augmented
        
        # Whitespace normalization
        if self.config.whitespace_normalization:
            normalized = self._normalize_whitespace(text)
            if normalized != text:
                new_example = deepcopy(example)
                new_example["text"] = normalized
                new_example["_augmented"] = True
                augmented.append(new_example)
        
        # Case variations (if enabled)
        if self.config.case_variations:
            # Add sentence-case version
            sentence_case = self._to_sentence_case(text)
            if sentence_case != text:
                new_example = deepcopy(example)
                new_example["text"] = sentence_case
                new_example["_augmented"] = True
                if len(augmented) < max_versions:
                    augmented.append(new_example)
        
        return augmented[:max_versions]

    def _augment_prompt_completion(
        self,
        example: Dict[str, Any],
        max_versions: int,
    ) -> List[Dict[str, Any]]:
        """Augment prompt/completion format."""
        augmented = []
        prompt = example.get("prompt", "")
        
        if not prompt:
            return augmented
        
        # Apply paraphrase variations to prompt
        if self.config.use_paraphrase_templates:
            variations = self.prompt_manager.get_paraphrase_variations(
                prompt,
                num_variations=max_versions,
            )
            
            for var_prompt in variations[1:]:  # Skip original
                new_example = deepcopy(example)
                new_example["prompt"] = var_prompt
                new_example["_augmented"] = True
                augmented.append(new_example)
        
        return augmented[:max_versions]

    def _transform_instruction(self, instruction: str) -> str:
        """Apply rule-based transformation to instruction."""
        transformed = instruction.lower()
        
        # Apply transformation patterns
        for pattern, replacement in self._instruction_transforms:
            if re.match(pattern, transformed, re.IGNORECASE):
                transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
                break
        
        # Capitalize first letter
        if transformed:
            transformed = transformed[0].upper() + transformed[1:]
        
        return transformed

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _to_sentence_case(self, text: str) -> str:
        """Convert text to sentence case."""
        sentences = re.split(r'([.!?]+\s*)', text)
        result = []
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Sentence content
                part = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
            result.append(part)
        
        return ''.join(result)

    def _swap_question_words(self, text: str) -> str:
        """Swap question words with synonyms."""
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('?')
            if word_lower in self._question_synonyms:
                synonyms = self._question_synonyms[word_lower]
                replacement = self.rng.choice(synonyms)
                
                # Preserve original case
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                words[i] = replacement
                break  # Only swap one word per call
        
        return ' '.join(words)


def augment_batch(
    examples: List[Dict[str, Any]],
    config: AugmentationConfig,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Convenience function to augment a batch of examples.
    
    Args:
        examples: List of training examples.
        config: Augmentation configuration.
        seed: Random seed.
        
    Returns:
        Augmented examples list.
    """
    augmenter = DataAugmenter(config, seed=seed)
    return augmenter.augment_dataset(examples)
