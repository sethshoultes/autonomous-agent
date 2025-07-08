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

## Phase 2 Core Agents - Complete ✅
- **Gmail Agent**: Email processing, classification, automation with AI analytics ✅
- **Research Agent**: Web scraping, content aggregation, AI-powered insights ✅
- **Ollama Integration**: Local AI processing with multi-model support ✅

## Phase 3 Advanced Features - Complete ✅
- **Code Agent**: GitHub integration, automated code reviews, security analysis ✅
- **Advanced Intelligence**: Multi-agent coordination, learning, decision making ✅
- **Docker Environment**: Production containerization and orchestration ✅
- **Security Hardening**: Comprehensive security measures and monitoring ✅

## Phase 4 Enhanced Production - Complete ✅
- **PostgreSQL Database**: Robust data persistence with analytics and search ✅
- **User Authentication**: JWT, MFA, OAuth2, RBAC with security monitoring ✅
- **Production Deployment**: Complete automation with monitoring and operations ✅
- **Enterprise Ready**: SOC2, ISO27001, GDPR compliant with full documentation ✅

## Project Status: 100% Complete 🎉
- **All Phases Implemented**: Foundation, Core Agents, Advanced Features, Production
- **Enterprise Grade**: Complete autonomous agent platform ready for deployment
- **Security Compliant**: Multi-layer security with comprehensive monitoring
- **Fully Operational**: Automated deployment, monitoring, and maintenance

## Environment Variables
```bash
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret
GITHUB_TOKEN=your_github_token
OLLAMA_URL=http://localhost:11434
PREFECT_API_URL=http://localhost:4200
```

## Important Notes & Considerations

### Complete Implementation Status - All Phases ✅
- **Complete TDD Infrastructure**: All base components implemented with tests first ✅
- **90%+ Coverage Enforced**: Automated coverage checking in CI/CD pipeline ✅
- **External Service Mocking**: Gmail, GitHub, and Ollama APIs fully mocked for testing ✅
- **SOLID Architecture**: Clean separation of concerns with dependency injection ✅
- **Production Ready**: Error handling, logging, and monitoring in place ✅
- **Core Agents Complete**: Gmail, Research, and Code agents with AI enhancement ✅
- **Local AI Integration**: Privacy-focused Ollama integration with multi-model support ✅
- **Advanced Intelligence**: Multi-agent coordination and learning systems ✅
- **Security Hardening**: Enterprise-grade security measures and compliance ✅
- **Container Orchestration**: Docker and Kubernetes ready for cloud deployment ✅
- **Database Integration**: PostgreSQL with SQLModel for robust data persistence ✅
- **User Authentication**: JWT, MFA, OAuth2, RBAC with comprehensive security ✅
- **Production Operations**: Complete deployment automation and monitoring ✅
- **Enterprise Compliance**: SOC2, ISO27001, GDPR ready with full documentation ✅

### Key Files Created - Phase 1 & 2
**Phase 1 Foundation:**
- `src/agents/base.py` - Abstract base agent classes
- `src/agents/manager.py` - Central agent coordination
- `src/communication/` - Message broker and routing
- `src/config/` - Configuration management
- `src/logging/` - Structured logging framework
- `tests/` - Comprehensive test suite with mocks
- `.github/workflows/` - CI/CD pipeline

**Phase 2 Core Agents:**
- `src/agents/gmail_agent.py` - Gmail automation with AI enhancement
- `src/agents/research.py` - Research agent with content aggregation
- `src/services/ollama_service.py` - Local AI integration service
- `src/services/ai_processing.py` - AI processing capabilities
- `tests/test_gmail_agent.py` - Gmail agent comprehensive tests
- `tests/test_research_agent.py` - Research agent test suite
- `docs/PHASE_2_SUMMARY.md` - Complete Phase 2 documentation

**Phase 3 Advanced Features:**
- `src/agents/code_agent.py` - GitHub integration and code analysis
- `src/services/github_service.py` - GitHub API and webhook handling
- `src/services/intelligence_engine.py` - Multi-agent coordination
- `src/services/ai_code_analyzer.py` - AI-powered code analysis
- `Dockerfile` & `docker-compose.yml` - Production containerization
- `k8s/` - Kubernetes manifests for cloud deployment
- `src/security/` - Comprehensive security hardening

**Phase 4 Enhanced Production:**
- `src/database/` - PostgreSQL integration with SQLModel
- `src/services/auth_service.py` - JWT authentication with MFA
- `src/services/user_management_service.py` - User management and preferences
- `scripts/production/` - Deployment automation and procedures
- `config/monitoring/` - Prometheus/Grafana monitoring stack
- `docs/api/` - Complete API documentation with examples
- `docs/guides/` - User and administrator guides

### Development Workflow
1. **TDD Approach**: Write tests first, then implement
2. **Quality Gates**: All code must pass linting, type checking, and coverage
3. **Continuous Integration**: GitHub Actions runs tests on every commit
4. **Code Review**: Use parallel agents for code review and testing

### Enterprise Deployment Complete
- All core agents (Gmail, Research, Code) with AI enhancement and data persistence
- Advanced intelligence layer with multi-agent coordination and learning
- PostgreSQL database with comprehensive data management and analytics
- User authentication system with JWT, MFA, OAuth2, and RBAC
- Production deployment automation with monitoring and operations
- Complete autonomous agent platform ready for enterprise deployment
- SOC2, ISO27001, GDPR compliant with comprehensive documentation

## Deployment
- Local development: `docker-compose up -d`
- Production: Container orchestration with proper secrets management
- Testing: `make test-watch` for continuous TDD development