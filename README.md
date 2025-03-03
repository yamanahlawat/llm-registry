# LLM Registry

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

LLM Registry is a Python package that provides a unified interface for discovering and managing the capabilities of various Large Language Models (LLMs). It includes a robust API, a rich CLI, and supports synchronization between local and remote model registries.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Library Usage](#library-usage)
- [CLI Usage](#cli-usage)
- [Model Capabilities](#model-capabilities)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

## Overview

Manage and discover LLM model capabilities across multiple providers like OpenAI, Anthropic, and more in a centralized registry. Use this package to check model capabilities before initializing provider clients and to manage model metadata efficiently.

## Features

- **Unified API** for capability discovery and management
- **Multiple Providers**: Supports OpenAI, Anthropic, and others
- **Local and Remote Storage**: Synchronize your model registries
- **Rich Command-Line Interface (CLI)**
- **Dynamic Capability Management**: Easily add, update, and delete model data

## Installation

Install via pip:

```bash
pip install llm-registry
```

## Library Usage

Integrate the package in your Python projects by following these steps:

### Listing Models

```python
from llm_registry import CapabilityRegistry, Provider
registry = CapabilityRegistry()
models = registry.get_models()
for model in models:
    print(model)
```

### Retrieve a Specific Model's Capabilities

```python
model = registry.get_model(Provider.OPENAI, "gpt-4")
if model and model.supports_streaming:
    from openai import OpenAI  # Replace with actual OpenAI client import
    client = OpenAI()  # Initialize client with streaming enabled
    response = client.chat.completions.create(
        model=model.model_id,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
```

### Add a New Model Capability

```python
from llm_registry.utils import create_model_capability
new_model = create_model_capability(
    model_id="gpt-4",
    provider=Provider.OPENAI,
    model_family="GPT-4",
    input_cost=0.01,
    output_cost=0.03,
    context_window=8192,
    training_cutoff="2023-04",
    supports_streaming=True,
    supports_tools=True,
    supports_json_mode=True,
    supports_system_prompt=True
)

from llm_registry import CapabilityRepository
repo = CapabilityRepository()
repo.save_model_capabilities(new_model)
```

## CLI Usage

The CLI tool `llmr` allows you to interact with model capabilities directly from the terminal.

### List Models

View all available models:

```bash
llmr list
```

![CLI Screenshot](./assets/images/cli.png)

*The above screenshot demonstrates how the CLI tool (`llmr`) currently looks like when listing models.

To filter models by provider:

```bash
llmr list --provider openai
```

### Add Model

Add a new model:

```bash
llmr add gpt-4 \
    --provider openai \
    --model-family GPT-4 \
    --input-cost 0.01 \
    --output-cost 0.03 \
    --context-window 8192 \
    --training-cutoff 2023-04 \
    --streaming \
    --tools \
    --json-mode \
    --system-prompt
```

### Delete Model

Remove an existing model:

```bash
llmr delete gpt-4 --provider openai
```

Use `-f` or `--force` to bypass confirmation.

## Model Capabilities

Each model entry tracks:

- **Basic Information**
  - Provider (e.g., OpenAI, Anthropic)
  - Model ID and Model Family
- **Cost Details**
  - Input/Output token costs (per 1M tokens)
  - Context window size
  - Training data cutoff date
- **Feature Support**
  - Streaming responses
  - Tools/Function calling
  - Vision/Image input
  - JSON mode
  - System prompt support

## Configuration

Default model data is stored in `~/.llm-registry`. You can override the directory by:
- Passing a `data_dir` parameter to `CapabilityRepository` in code
- Using the `--data-dir` option in CLI commands

## Development

### Requirements
- Python 3.13+
- [uv](https://github.com/your_org/uv) for dependency management

### Setup

```bash
# Create virtual environment and sync dependencies
uv venv
uv sync --group dev

# Run tests with coverage analysis
pytest -v --cov=llm_registry
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
