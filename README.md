<div align="center">

# 🤖 LLM Registry

*Your Central Hub for LLM Model Management*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)]()

</div>

---

LLM Registry is a Python package that provides a unified interface for discovering and managing the capabilities of various Large Language Models (LLMs). It includes a robust API, a rich CLI, and supports both package-included and user-managed model registries with local storage. The package supports multi-provider models, allowing a single model to be associated with multiple providers.

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

## <a id="overview"></a>🎯 Overview

Manage and discover LLM model capabilities across multiple providers like OpenAI, Anthropic, and more in a centralized registry. Use this package to check model capabilities before initializing provider clients and to manage model metadata efficiently.

> 💡 Perfect for teams managing multiple LLM providers and wanting to standardize their model interactions.

## <a id="features"></a>✨ Features

🔗 **Unified API**
- Single interface for capability discovery and management
- Consistent experience across all providers

🏢 **Multiple Providers**
- Support for OpenAI, Anthropic, Google, Cohere, Mistral, Meta, and more
- Multi-provider model support - associate a single model with multiple providers

💾 **Smart Storage**
- Local storage for model metadata
- Package-included and user-managed registries
- Efficient caching mechanism

🖥️ **Rich CLI Experience**
- Intuitive commands for model management
- Beautiful terminal output with rich formatting
- Quick access to model information

📊 **Comprehensive Tracking**
- Token costs (standard and cached)
- Model features and capabilities
- API parameters and limitations
- Training cutoff dates

🗂️ **Organization**
- Group models by family and provider
- Easy filtering and search

## <a id="installation"></a>🚀 Installation

Install via uv:

```bash
uv add llm-registry
```

---

Install via pip:

```bash
pip install llm-registry
```

## <a id="library-usage"></a>📚 Library Usage

Integrate the package in your Python projects by following these steps:

### 📋 Listing Models

```python
from llm_registry import CapabilityRegistry, Provider
registry = CapabilityRegistry()
models = registry.get_models()
for model in models:
    print(model)
```

### 🔍 Retrieve a Specific Model's Capabilities

```python
model = registry.get_model("gpt-4")
if model and model.api_params.stream:
    from openai import OpenAI  # Replace with actual OpenAI client import
    client = OpenAI()  # Initialize client with streaming enabled
    response = client.chat.completions.create(
        model=model.model_id,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
```

### ➕ Add a New Model Capability

```python
from llm_registry.utils import create_model_capability
from llm_registry import Provider

# Single provider model
new_model = create_model_capability(
    model_id="gpt-4",
    provider=Provider.OPENAI,  # Automatically converted to list internally
    model_family="GPT-4",
    input_cost=0.01,
    output_cost=0.03,
    cache_input_cost=0.005,  # Optional cached token costs
    cache_output_cost=0.015,
    context_window=8192,
    training_cutoff="2023-04",
    supports_streaming=True,
    supports_tools=True,
    supports_json_mode=True,
    supports_system_prompt=True
)

# Multi-provider model
multi_provider_model = create_model_capability(
    model_id="llama-2-70b",
    provider=[Provider.META, Provider.GITHUB],
    model_family="Llama 2",
    input_cost=0.0007,
    output_cost=0.0009,
    context_window=4096
)

from llm_registry import CapabilityRepository
repo = CapabilityRepository()
repo.save_model_capabilities(new_model)
repo.save_model_capabilities(multi_provider_model)
```

## <a id="cli-usage"></a>💻 CLI Usage

The CLI tools `llmr` and `llm-registry` allow you to interact with model capabilities directly from the terminal.

### 📋 List Models

View all available models:

```bash
llmr list
```

![CLI Screenshot](./assets/images/cli.png)

*The above screenshot demonstrates how the CLI tool (`llmr`) currently looks like when listing models.*

To filter models by provider:

```bash
llmr list --provider openai
```

Additional filtering options:

```bash
# Show only user-defined models
llmr list --user-only

# Show only package-included models
llmr list --package-only
```

### 🔍 Get Detailed Model Information

Get detailed information about a specific model:

```bash
llmr get gpt-4
```

For JSON output:

```bash
llmr get gpt-4 --json
```

Output
```
❯ llmr get o1 --json
{
  "model_id": "o1",
  "providers": [
    "openai"
  ],
  "model_family": "o1",
  "base_model": null,
  "api_params": {
    "max_tokens": true,
    "temperature": false,
    "top_p": false,
    "stream": true
  },
  "features": {
    "vision": true,
    "tools": true,
    "json_mode": true,
    "system_prompt": false
  },
  "token_costs": {
    "input_cost": 15.0,
    "output_cost": 60.0,
    "cache_input_cost": 7.5,
    "cache_output_cost": null,
    "context_window": 200000,
    "training_cutoff": "2023-10"
  }
}
```

### ➕ Add Model

Add a new model:

```bash
llmr add gpt-4 \
    --provider openai \
    --model-family GPT-4 \
    --input-cost 0.01 \
    --output-cost 0.03 \
    --cache-input-cost 0.005 \
    --cache-output-cost 0.015 \
    --context-window 8192 \
    --training-cutoff 2023-04 \
    --stream \
    --tools \
    --json-mode \
    --system-prompt
```

### 🔄 Update Model

Update an existing model:

```bash
llmr update gpt-4 \
    --provider openai \
    --model-family "GPT-4 Turbo" \
    --input-cost 0.005 \
    --output-cost 0.015 \
    --vision
```

### 🗑️ Delete Model

Remove an existing model:

```bash
llmr delete gpt-4 --provider openai
```

Use `-f` or `--force` to bypass confirmation.

## <a id="model-capabilities"></a>🎯 Model Capabilities

Each model entry tracks:

🏷️ **Basic Information**
  - Providers (supports multiple providers per model)
  - Model ID and Model Family

💰 **Cost Details**
  - Input/Output token costs (per 1M tokens)
  - Cached Input/Output token costs (per 1M tokens)
  - Context window size
  - Training data cutoff date

⚙️ **API Parameters**
  - Max tokens support
  - Temperature support
  - Top-p support
  - Streaming support

✨ **Feature Support**
  - Streaming responses
  - Tools/Function calling
  - Vision/Image input
  - JSON mode
  - System prompt support

---

## 👥 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

---

## <a id="configuration"></a>Configuration

Default model data is stored in `~/.llm-registry`. You can override the directory by:
- Passing a `data_dir` parameter to `CapabilityRepository` in code
- Using the `--data-dir` option in CLI commands

## <a id="development"></a>Development

### Requirements
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
# Create virtual environment and sync dependencies
uv venv
uv sync --group dev

# Run tests with coverage analysis
pytest -v --cov=llm_registry
```

## <a id="license"></a>License

Distributed under the MIT License. See `LICENSE` for more information.
