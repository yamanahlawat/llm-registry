# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
