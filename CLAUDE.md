# Claude Assistant Configuration

## Project Overview
Autonomous agent system for email management, research, code review, and development tasks.

## Development Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Testing (TDD Approach)
pytest tests/ -v
python -m pytest --cov=src --cov-report=html
python -m pytest --cov=src --cov-fail-under=90

# Linting & Type Checking
ruff check src/
ruff format src/
mypy src/

# Running
python -m src.main
docker-compose up -d
```

## Architecture Principles
- **KISS**: Simple, straightforward solutions
- **DRY**: No code duplication
- **YAGNI**: Build only what's needed
- **SOLID**: Well-structured, maintainable code
- **Zen of Python**: Pythonic, readable code

## Key Components
- **Orchestrator**: Prefect-based workflow management
- **Gmail Agent**: Email processing and automation
- **Research Agent**: Web scraping and content aggregation
- **Code Agent**: GitHub integration and review automation
- **Ollama Integration**: Local AI processing

## Environment Variables
```bash
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret
GITHUB_TOKEN=your_github_token
OLLAMA_URL=http://localhost:11434
PREFECT_API_URL=http://localhost:4200
```

## Deployment
- Local development: `docker-compose up -d`
- Production: Container orchestration with proper secrets management