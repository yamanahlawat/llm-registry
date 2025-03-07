# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
