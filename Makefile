# Autonomous Agent Testing Makefile
# 
# This Makefile provides convenient commands for running tests, linting,
# and other development tasks following TDD principles.

.PHONY: help install test test-unit test-integration test-all lint format type-check security coverage clean docs

# Default target
help:
	@echo "Autonomous Agent Testing Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  install           Install all dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo ""
	@echo "Testing (TDD Workflow):"
	@echo "  test              Run all tests (unit + integration)"
	@echo "  test-unit         Run unit tests only (fast)"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-watch        Run tests in watch mode (for TDD)"
	@echo "  test-coverage     Run tests with detailed coverage report"
	@echo "  test-performance  Run performance tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run all linting checks"
	@echo "  format            Format code with black and ruff"
	@echo "  type-check        Run mypy type checking"
	@echo "  security          Run security checks"
	@echo ""
	@echo "Reports:"
	@echo "  coverage          Generate coverage report"
	@echo "  coverage-html     Generate HTML coverage report"
	@echo "  docs              Build documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean             Clean up build artifacts and cache"
	@echo "  update-deps       Update dependencies"
	@echo "  pre-commit        Run pre-commit hooks"

# ============================================================================
# Setup Commands
# ============================================================================

install:
	@echo "Installing dependencies..."
	poetry install

install-dev:
	@echo "Installing development dependencies..."
	poetry install --with dev,test

# ============================================================================
# Testing Commands (TDD Workflow)
# ============================================================================

test:
	@echo "Running all tests..."
	poetry run pytest tests/ -v

test-unit:
	@echo "Running unit tests (TDD Red-Green-Refactor)..."
	poetry run pytest tests/unit/ -v -m "unit" --tb=short

test-integration:
	@echo "Running integration tests..."
	poetry run pytest tests/integration/ -v -m "integration" --tb=short

test-watch:
	@echo "Running tests in watch mode for TDD workflow..."
	poetry run pytest-watch tests/ -- -v --tb=short

test-coverage:
	@echo "Running tests with coverage analysis..."
	poetry run pytest tests/ \
		--cov=src \
		--cov-report=html:htmlcov \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=90 \
		-v

test-performance:
	@echo "Running performance tests..."
	poetry run pytest tests/ -m "performance" -v --tb=short

test-security:
	@echo "Running security tests..."
	poetry run pytest tests/ -m "security" -v --tb=short

test-slow:
	@echo "Running slow tests..."
	poetry run pytest tests/ -m "slow" -v --tb=short

test-smoke:
	@echo "Running smoke tests..."
	poetry run pytest tests/ -m "smoke" -v --tb=short

test-parallel:
	@echo "Running tests in parallel..."
	poetry run pytest tests/ -n auto -v

test-failed:
	@echo "Re-running failed tests..."
	poetry run pytest --lf -v

test-verbose:
	@echo "Running tests with maximum verbosity..."
	poetry run pytest tests/ -vvv --tb=long

# ============================================================================
# Code Quality Commands
# ============================================================================

lint:
	@echo "Running all linting checks..."
	@make lint-ruff
	@make lint-mypy
	@make lint-bandit

lint-ruff:
	@echo "Running ruff linter..."
	poetry run ruff check src tests

lint-mypy:
	@echo "Running mypy type checker..."
	poetry run mypy src tests

lint-bandit:
	@echo "Running bandit security linter..."
	poetry run bandit -r src/ -f json -o bandit-report.json || true
	poetry run bandit -r src/

format:
	@echo "Formatting code..."
	poetry run ruff format src tests
	poetry run ruff check --fix src tests
	poetry run isort src tests

format-check:
	@echo "Checking code formatting..."
	poetry run ruff format --check src tests
	poetry run isort --check-only src tests

type-check:
	@echo "Running type checking..."
	poetry run mypy src tests

security:
	@echo "Running security checks..."
	poetry run safety check
	poetry run bandit -r src/

security-audit:
	@echo "Running comprehensive security audit..."
	poetry run safety check --json --output safety-report.json || true
	poetry run bandit -r src/ -f json -o bandit-report.json || true
	@echo "Security reports generated: safety-report.json, bandit-report.json"

# ============================================================================
# Coverage Commands
# ============================================================================

coverage:
	@echo "Generating coverage report..."
	poetry run pytest tests/ --cov=src --cov-report=term-missing

coverage-html:
	@echo "Generating HTML coverage report..."
	poetry run pytest tests/ --cov=src --cov-report=html:htmlcov
	@echo "Coverage report available at: htmlcov/index.html"

coverage-xml:
	@echo "Generating XML coverage report..."
	poetry run pytest tests/ --cov=src --cov-report=xml:coverage.xml

coverage-json:
	@echo "Generating JSON coverage report..."
	poetry run pytest tests/ --cov=src --cov-report=json:coverage.json

coverage-lcov:
	@echo "Generating LCOV coverage report..."
	poetry run pytest tests/ --cov=src --cov-report=lcov:coverage.lcov

# ============================================================================
# Documentation Commands
# ============================================================================

docs:
	@echo "Building documentation..."
	poetry run sphinx-build -b html docs/ docs/_build/html/
	@echo "Documentation available at: docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf docs/_build/

docs-watch:
	@echo "Building documentation in watch mode..."
	poetry run sphinx-autobuild docs/ docs/_build/html/

docs-linkcheck:
	@echo "Checking documentation links..."
	poetry run sphinx-build -b linkcheck docs/ docs/_build/linkcheck/

# ============================================================================
# Development Commands
# ============================================================================

dev-setup:
	@echo "Setting up development environment..."
	@make install-dev
	@make pre-commit-install
	@echo "Development environment ready!"

pre-commit:
	@echo "Running pre-commit hooks..."
	poetry run pre-commit run --all-files

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	poetry run pre-commit install

pre-commit-update:
	@echo "Updating pre-commit hooks..."
	poetry run pre-commit autoupdate

# ============================================================================
# Maintenance Commands
# ============================================================================

clean:
	@echo "Cleaning up build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	find . -type f -name "*.rej" -delete

clean-coverage:
	@echo "Cleaning coverage reports..."
	rm -rf htmlcov/
	rm -f coverage.xml coverage.json coverage.lcov .coverage

clean-docs:
	@echo "Cleaning documentation..."
	rm -rf docs/_build/

clean-all: clean clean-coverage clean-docs
	@echo "All clean!"

update-deps:
	@echo "Updating dependencies..."
	poetry update
	poetry show --outdated

update-deps-dev:
	@echo "Updating development dependencies..."
	poetry update --with dev,test

# ============================================================================
# CI/CD Commands
# ============================================================================

ci:
	@echo "Running CI pipeline locally..."
	@make lint
	@make test-coverage
	@make security
	@make docs

ci-fast:
	@echo "Running fast CI checks..."
	@make lint-ruff
	@make test-unit

ci-full:
	@echo "Running full CI pipeline..."
	@make lint
	@make test-all
	@make security-audit
	@make docs

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t autonomous-agent:latest .

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm autonomous-agent:latest make test

docker-lint:
	@echo "Running linting in Docker..."
	docker run --rm autonomous-agent:latest make lint

# ============================================================================
# Benchmark Commands
# ============================================================================

benchmark:
	@echo "Running performance benchmarks..."
	poetry run pytest tests/ --benchmark-only --benchmark-sort=mean

benchmark-save:
	@echo "Running and saving benchmarks..."
	poetry run pytest tests/ --benchmark-only --benchmark-save=baseline

benchmark-compare:
	@echo "Comparing benchmarks..."
	poetry run pytest tests/ --benchmark-only --benchmark-compare=baseline

# ============================================================================
# Profiling Commands
# ============================================================================

profile:
	@echo "Running memory profiler..."
	poetry run python -m memory_profiler src/main.py

profile-line:
	@echo "Running line profiler..."
	poetry run kernprof -l -v src/main.py

# ============================================================================
# Database Commands (if applicable)
# ============================================================================

db-migrate:
	@echo "Running database migrations..."
	poetry run alembic upgrade head

db-rollback:
	@echo "Rolling back database migration..."
	poetry run alembic downgrade -1

db-reset:
	@echo "Resetting test database..."
	# Add database reset commands here

# ============================================================================
# Special Testing Scenarios
# ============================================================================

test-tdd:
	@echo "Running TDD workflow (Red-Green-Refactor)..."
	@echo "1. RED: Running tests (should fail initially)..."
	poetry run pytest tests/unit/ -x --tb=short || true
	@echo ""
	@echo "2. GREEN: Implement minimal code to pass tests"
	@echo "3. REFACTOR: Improve code while keeping tests green"
	@echo ""
	@echo "Use 'make test-watch' for continuous TDD feedback"

test-bdd:
	@echo "Running BDD-style tests..."
	poetry run pytest tests/ -v --tb=short --gherkin-terminal-reporter

test-mutation:
	@echo "Running mutation testing..."
	poetry run mutmut run --paths-to-mutate=src/

test-property:
	@echo "Running property-based tests..."
	poetry run pytest tests/ -m "hypothesis" -v

# ============================================================================
# Reporting Commands
# ============================================================================

report:
	@echo "Generating comprehensive test report..."
	@make test-coverage
	@make security-audit
	@make docs
	@echo "Reports generated:"
	@echo "- Coverage: htmlcov/index.html"
	@echo "- Security: bandit-report.json, safety-report.json"
	@echo "- Documentation: docs/_build/html/index.html"

report-json:
	@echo "Generating JSON reports..."
	poetry run pytest tests/ \
		--cov=src \
		--cov-report=json:coverage.json \
		--json-report --json-report-file=test-report.json

# ============================================================================
# Example TDD Workflow
# ============================================================================

tdd-example:
	@echo "TDD Example Workflow:"
	@echo "1. RED:   Write a failing test"
	@echo "2. GREEN: Write minimal code to pass the test"
	@echo "3. REFACTOR: Improve the code while keeping tests green"
	@echo ""
	@echo "Commands for TDD workflow:"
	@echo "  make test-watch    # Continuous test running"
	@echo "  make test-unit     # Run unit tests quickly"
	@echo "  make test-failed   # Re-run only failed tests"
	@echo "  make format        # Clean up code after refactoring"

# ============================================================================
# Targets that don't create files
# ============================================================================

.PHONY: help install install-dev test test-unit test-integration test-all \
        test-watch test-coverage test-performance test-security test-slow \
        test-smoke test-parallel test-failed test-verbose lint lint-ruff \
        lint-mypy lint-bandit format format-check type-check security \
        security-audit coverage coverage-html coverage-xml coverage-json \
        coverage-lcov docs docs-clean docs-watch docs-linkcheck dev-setup \
        pre-commit pre-commit-install pre-commit-update clean clean-coverage \
        clean-docs clean-all update-deps update-deps-dev ci ci-fast ci-full \
        docker-build docker-test docker-lint benchmark benchmark-save \
        benchmark-compare profile profile-line db-migrate db-rollback \
        db-reset test-tdd test-bdd test-mutation test-property report \
        report-json tdd-example