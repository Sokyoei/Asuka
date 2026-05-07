# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sokyoei's AI/LLM learning and research project containing:

- **Ahri/Asuka/**: Python package
- **include/Ahri/Asuka/**: C/C++ includes
- **src/Ahri/Asuka/** : C/C++ source files
- **XXX_learn/** or **learning/** : Learning projects

**Migration Strategy**: Learning projects are experimental. Code in `learning/` and `XXX_learn/` directories may be migrated to main package when useful:

- C++ code -> `include/Ahri/Asuka/` or `src/Ahri/Asuka/`, try use header only
- Python code -> `Ahri/Asuka/` modules

## Environments

- Python: Use `uv` in `conda` environment, use `uv pip` install packages.
- C++: Use `vcpkg` global mode.

## Code Styles

### Comments

- **Chinese/English mix**: When using Chinese with English, add space between them (e.g., "我的 TensorRT 模型")
- **Large blocks**: Use continuous separators with the same comment character (e.g., `#` for Python, `//` for C++)
- **Language choice**: Use comments in the current language

### Python

- Use Google docstrings and follow ruff style (see `pyproject.toml`)
- Use type hints and follow PEP 484
- Use `black` for formatting and follow PEP 8 (see `pyproject.toml`)

### C++

- Use Doxygen comments (e.g., `@brief`, `@param`, `@return`)
- Follow Chromium style formatting (see `.clang-format`)

## Documentation

Project uses `MkDocs with Material` theme (Chinese documentation). Math formulas use LaTeX. Markdown files in `docs/` directory.
