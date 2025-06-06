# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.8] - 2025-06-06

### Changed

- **Project Configuration**: Updated project.urls in pyproject.toml
  - Homepage: Set to "https://github.com/yamanahlawat/llm-registry"
  - Issues: Set to "https://github.com/yamanahlawat/llm-registry/issues"
- **Maintenance:**
  - Updated dependencies to latest minor versions.

## [0.3.7] - 2025-06-02

### 0.3.7 Highlights - Comprehensive Model Validation & Corrections

**Fixed:**

- **Training Cutoff Dates**: Corrected training cutoff dates for multiple models that had future dates:
  - o3-mini: "2023-10" → "2024-10"
  - Claude 4 models: "2025-05" → "2025-03"
  - Phi-4-reasoning: "2025-05" → "2025-03"
  - DeepSeek models: "2025-05" → "2024-12"
  - Mistral-medium-3: "2025-05" → "2025-02"
  - Qwen3: "2025-04" → "2025-02"

- **Model Specifications**: Updated specifications based on verified official sources:
  - Phi-4-reasoning context window: 128000 → 32000 tokens
  - Mistral Medium 3 pricing: Updated to official $0.40/$20.80 per million tokens
  - DeepSeek provider consistency: Standardized provider to "deepseek"

- **Cache Costs**: Added missing cache_input_cost values to GPT-4.1 family models for consistency

**Added:**

- **Provider Support**: Added "microsoft" and "deepseek" to Provider enum in models.py
- **Model Existence Verification**: Conducted comprehensive web research to verify all newly added models exist:
  - OpenAI o3/o4-mini models (confirmed released April 16, 2025)
  - Claude 4 Opus/Sonnet (confirmed released May 23, 2025)
  - Microsoft Phi-4 reasoning models (confirmed released May 1, 2025)
  - DeepSeek R1 (confirmed released January 20, 2025)
  - Mistral Medium 3 (confirmed released May 7, 2025)
  - Qwen3 (confirmed released April 2025)

## [0.3.6] - 2025-05-24

### 0.3.6 Highlights

**Added:**

- Added latest OpenAI models: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano with accurate pricing, features, and context window.
- Added new Anthropic Claude 4 models: claude-opus-4, claude-sonnet-4 with up-to-date pricing and features.
- Added trending open-source models from Ollama: llama-4, gemma3, qwen3, devstral, qwen2.5vl.

**Changed:**

- Grouped OpenAI and Claude models together in models.json for better organization.
- Updated all model entries for gpt-4.1 and Claude 4 series with 100% accurate cost, features, and context window based on latest sources.
- Updated Ollama model entries to reflect correct API parameters and features (set json_mode and system_prompt to false where not supported).

**Maintenance:**

- Validated and deduplicated models.json for correctness and completeness.

## [0.3.5] - 2025-05-20

### Added

- Fix llmr list command.

### Fixed

- Fixed minor typos in CLI error messages.

### Maintenance

- Updated dependencies to latest minor versions.

## [0.3.4] - 2025-05-20

### Added

- Added new llms models config.

### Update

- models.json deduplicated and validated for correctness.

### Enhancement

- Ensured all model entries are unique and up-to-date.

### Maintenance

- JSON structure checked and confirmed valid.

- Version bump to 0.3.4 for new release.
- Rebuilt wheels and sdist for PyPI publishing.

## [0.3.3] - 2025-03-23

- Refactor: Readme Table content navigation.
- Added support for partial match for ex. check deepseek-r1  for deepseek-r1:8b, deepseek-r1:14b.
- Added Mistral, llama3.3, qwen2.5-coder models.

## [0.3.2] - 2025-03-07

### Added
- New models:
  - Deepseek series: deepseek-r1 (via GitHub)
  - Phi-4 series: phi-4 and phi-4-multimodal-instruct (via GitHub)
    - phi-4: Base model with JSON mode and system prompt support
    - phi-4-multimodal-instruct: Enhanced version with vision and tools support

### Changed
- Enhanced error handling in CLI
  - Introduced ModelNotFoundError for better error reporting
  - Updated get command to use new exception type
- Model capabilities:
  - Full API parameter support (max_tokens, temperature, top_p, stream)
  - Advanced features in phi-4-multimodal-instruct (vision, tools, json_mode, system_prompt)

## [0.3.1] - 2025-03-06
- Updated model api_params, features, and token_costs

## [0.3.0] - 2025-03-04

### Added
- Added validation for cache costs in TokenCost model
- Enhanced error handling for corrupted JSON files
- Comprehensive test suite with 98% coverage
- Added pytest configuration
- New test files: test_integration.py, test_models.py, test_utils.py

### Changed
- Simplified ApiParams by removing rarely used parameters
- Improved file handling with proper UTF-8 encoding
- Enhanced model loading caching mechanism
- Restructured data directory management
- Updated gitignore with comprehensive Python patterns
- Improved README documentation with coverage badge

### Fixed
- Fixed default value handling in Features class
- Added proper data structure validation in save_user_models
- Improved error handling for corrupted JSON files
- Excluded models.json from formatting to preserve provider organization

## [0.2.0] - 2025-03-04

### Added
- Support for caching input and output token costs
- Comprehensive API parameter support (max_tokens, temperature, top_p, etc.)
- Model grouping functionality
- Cache mechanism for loading models file
- Added development dependencies: ipython and ipdb

### Changed
- Restructured CLI to use centralized model data management
- Enhanced model listing with additional cost columns
- Improved model capabilities creation with separate API params and features

## [0.1.0] - 2025-03-03

### Added
- Initial release
- Core functionality for managing LLM model capabilities
- Support for OpenAI and Anthropic models
- CLI interface with list, add, and delete commands
- Local storage for model capabilities
- Rich table output for model listing
- Comprehensive test suite
- MIT License
- Detailed documentation in README.md
