# Claude Assistant Configuration

## Project Overview
Autonomous agent system for email management, research, code review, and development tasks.

## Development Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Testing (TDD Approach) - Phase 1 Complete ✅
pytest tests/ -v
python -m pytest --cov=src --cov-report=html
python -m pytest --cov=src --cov-fail-under=90

# Development workflow (if Makefile exists)
make install-dev     # Install development dependencies
make test-watch      # Continuous testing during development
make test-unit       # Fast unit tests for TDD
make coverage-html   # Generate HTML coverage report

# Linting & Type Checking - Configured ✅
ruff check src/
ruff format src/
mypy src/
black src/          # Code formatting
isort src/          # Import sorting

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
- **TDD**: Test Driven Development with 90%+ coverage ✅

## Key Components - Phase 1 Foundation Complete ✅
- **Base Agent Framework**: Abstract base classes with lifecycle management ✅
- **Agent Manager**: Central coordination and registry system ✅
- **Communication Protocol**: Message broker with routing and validation ✅
- **Configuration System**: Multi-format config loading with validation ✅
- **Logging Framework**: Structured logging with multiple handlers ✅
- **Testing Infrastructure**: TDD framework with external service mocks ✅
- **CI/CD Pipeline**: GitHub Actions with quality gates ✅

## Phase 2 Components (Next)
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

## Important Notes & Considerations

### Phase 1 Implementation Status
- **Complete TDD Infrastructure**: All base components implemented with tests first
- **90%+ Coverage Enforced**: Automated coverage checking in CI/CD pipeline
- **External Service Mocking**: Gmail, GitHub, and Ollama APIs fully mocked for testing
- **SOLID Architecture**: Clean separation of concerns with dependency injection
- **Production Ready**: Error handling, logging, and monitoring in place

### Key Files Created
- `src/agents/base.py` - Abstract base agent classes
- `src/agents/manager.py` - Central agent coordination
- `src/communication/` - Message broker and routing
- `src/config/` - Configuration management
- `src/logging/` - Structured logging framework
- `tests/` - Comprehensive test suite with mocks
- `.github/workflows/` - CI/CD pipeline

### Development Workflow
1. **TDD Approach**: Write tests first, then implement
2. **Quality Gates**: All code must pass linting, type checking, and coverage
3. **Continuous Integration**: GitHub Actions runs tests on every commit
4. **Code Review**: Use parallel agents for code review and testing

### Next Phase Preparation
- Base framework ready for Gmail, Research, and Code agents
- External service integrations prepared with proper mocking
- Configuration system ready for API credentials
- Logging and monitoring infrastructure in place

## Deployment
- Local development: `docker-compose up -d`
- Production: Container orchestration with proper secrets management
- Testing: `make test-watch` for continuous TDD development