# Autonomous Agent System

A privacy-focused autonomous agent for email management, research, and development tasks using local AI processing.

## Features

- **Email Automation**: Gmail processing, classification, and automated responses
- **Research Agent**: Web scraping and content aggregation
- **Code Agent**: GitHub integration and automated code reviews
- **Local AI**: Ollama integration for privacy-focused processing
- **Orchestration**: Prefect-based workflow management

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings

# Run
python -m src.main
```

## Architecture

Built following KISS, DRY, YAGNI, SOLID principles and Test Driven Development.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Development

See [CLAUDE.md](CLAUDE.md) for development commands and configuration.

## License

MIT License - see LICENSE file for details.