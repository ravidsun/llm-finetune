#!/usr/bin/env python3
"""
Create sample JSONL training data for testing
"""
import json
import argparse
from pathlib import Path


SAMPLE_DATA = [
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention."
    },
    {
        "instruction": "Explain neural networks",
        "input": "",
        "output": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process and transform input data to produce outputs. Each connection has a weight that adjusts as learning proceeds, allowing the network to learn patterns."
    },
    {
        "instruction": "What is deep learning?",
        "input": "",
        "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to progressively extract higher-level features from raw input. It excels at tasks like image recognition, natural language processing, and speech recognition by learning hierarchical representations."
    },
    {
        "instruction": "Define supervised learning",
        "input": "",
        "output": "Supervised learning is a machine learning approach where the algorithm learns from labeled training data. The model is trained on input-output pairs and learns to map inputs to correct outputs, which it can then apply to new, unseen data. Examples include classification and regression tasks."
    },
    {
        "instruction": "What is reinforcement learning?",
        "input": "",
        "output": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time through trial and error, without explicit supervision."
    },
    {
        "instruction": "Explain gradient descent",
        "input": "",
        "output": "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It iteratively adjusts model parameters in the direction of steepest descent of the loss function, gradually moving towards the optimal parameter values that minimize prediction errors."
    },
    {
        "instruction": "What is overfitting?",
        "input": "",
        "output": "Overfitting occurs when a machine learning model learns the training data too well, including its noise and outliers, resulting in poor performance on new, unseen data. The model becomes too complex and fails to generalize, essentially memorizing rather than learning underlying patterns."
    },
    {
        "instruction": "Define cross-validation",
        "input": "",
        "output": "Cross-validation is a resampling technique used to evaluate machine learning models by dividing data into multiple subsets. The model is trained on some subsets and validated on others, rotating through all combinations. This provides a more robust estimate of model performance and helps detect overfitting."
    },
    {
        "instruction": "What is a transformer?",
        "input": "",
        "output": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data. Introduced in the 'Attention is All You Need' paper, transformers have become the foundation of modern language models like GPT and BERT, excelling at capturing long-range dependencies in text."
    },
    {
        "instruction": "Explain transfer learning",
        "input": "",
        "output": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It leverages knowledge gained from solving one problem to solve a related problem, often requiring less data and training time for the new task."
    },
]


def create_sample_data(output_path: str, num_samples: int = 100):
    """Create sample JSONL training data"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Repeat sample data to reach desired number of samples
    samples = []
    while len(samples) < num_samples:
        samples.extend(SAMPLE_DATA)

    samples = samples[:num_samples]

    # Write to JSONL
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✅ Created {num_samples} sample training examples")
    print(f"📁 Saved to: {output_file}")
    print(f"\nPreview (first 2 lines):")

    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            print(f"  {json.loads(line)}")


def main():
    parser = argparse.ArgumentParser(description="Create sample JSONL training data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/input/train.jsonl",
        help="Output JSONL file path (default: data/input/train.jsonl)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )

    args = parser.parse_args()
    create_sample_data(args.output, args.samples)


if __name__ == "__main__":
    main()
