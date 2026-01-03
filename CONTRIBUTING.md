# Contributing to LLM Fine-Tuning Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, GPU type)
- Relevant logs or error messages

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

#### Pull Request Guidelines

- Follow the existing code style
- Add tests for new features
- Update the README if you change functionality
- Keep commits focused and atomic
- Write clear commit messages

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-finetune.git
cd llm-finetune

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=finetune_project tests/
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: Add support for Mistral model fine-tuning

- Add Mistral model configuration
- Update prompt templates for Mistral format
- Add example config file

Closes #123
```

## Questions?

Feel free to open an issue for any questions about contributing!
