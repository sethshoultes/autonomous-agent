#!/bin/bash

# ============================================================================
# Setup Script for Autonomous Agent System
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.11"
NODE_VERSION="18"

# Function to log messages
log_message() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists pacman; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    local os=$(detect_os)
    
    log_message "Installing system dependencies for $os..."
    
    case $os in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv python3-dev \
                postgresql postgresql-contrib \
                redis-server \
                nginx \
                curl wget git \
                build-essential \
                libpq-dev \
                pkg-config \
                openssl \
                docker.io docker-compose \
                nodejs npm
            ;;
        centos)
            sudo yum update -y
            sudo yum install -y \
                python3 python3-pip python3-devel \
                postgresql postgresql-server postgresql-contrib \
                redis \
                nginx \
                curl wget git \
                gcc gcc-c++ make \
                libpq-devel \
                openssl-devel \
                docker docker-compose \
                nodejs npm
            ;;
        macos)
            if ! command_exists brew; then
                log_message "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                python@$PYTHON_VERSION \
                postgresql \
                redis \
                nginx \
                curl wget git \
                openssl \
                docker docker-compose \
                node@$NODE_VERSION
            ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Function to setup Python environment
setup_python_environment() {
    log_message "Setting up Python environment..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Poetry
    pip install poetry
    
    # Install dependencies
    poetry install
    
    log_success "Python environment setup completed"
}

# Function to setup database
setup_database() {
    log_message "Setting up PostgreSQL database..."
    
    local os=$(detect_os)
    
    # Start PostgreSQL service
    case $os in
        ubuntu)
            sudo systemctl start postgresql
            sudo systemctl enable postgresql
            ;;
        centos)
            sudo systemctl start postgresql
            sudo systemctl enable postgresql
            ;;
        macos)
            brew services start postgresql
            ;;
    esac
    
    # Create database and user
    sudo -u postgres psql -c "CREATE USER agent WITH PASSWORD 'agent123';" || true
    sudo -u postgres psql -c "CREATE DATABASE autonomous_agent OWNER agent;" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE autonomous_agent TO agent;" || true
    
    # Run database initialization
    PGPASSWORD=agent123 psql -h localhost -U agent -d autonomous_agent -f "$PROJECT_DIR/docker/postgres/init.sql"
    
    log_success "Database setup completed"
}

# Function to setup Redis
setup_redis() {
    log_message "Setting up Redis..."
    
    local os=$(detect_os)
    
    # Start Redis service
    case $os in
        ubuntu)
            sudo systemctl start redis-server
            sudo systemctl enable redis-server
            ;;
        centos)
            sudo systemctl start redis
            sudo systemctl enable redis
            ;;
        macos)
            brew services start redis
            ;;
    esac
    
    log_success "Redis setup completed"
}

# Function to setup Docker
setup_docker() {
    log_message "Setting up Docker..."
    
    local os=$(detect_os)
    
    case $os in
        ubuntu)
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        centos)
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        macos)
            log_message "Please start Docker Desktop manually"
            ;;
    esac
    
    log_success "Docker setup completed"
}

# Function to setup monitoring
setup_monitoring() {
    log_message "Setting up monitoring stack..."
    
    # Create monitoring directories
    mkdir -p /opt/autonomous-agent/monitoring/{prometheus,grafana}
    
    # Copy monitoring configurations
    cp -r "$PROJECT_DIR/docker/prometheus" /opt/autonomous-agent/monitoring/
    cp -r "$PROJECT_DIR/docker/grafana" /opt/autonomous-agent/monitoring/
    
    log_success "Monitoring setup completed"
}

# Function to generate secrets
generate_secrets() {
    log_message "Generating secrets..."
    
    "$PROJECT_DIR/scripts/generate-secrets.sh" generate
    
    log_success "Secrets generated"
}

# Function to setup SSL certificates
setup_ssl() {
    log_message "Setting up SSL certificates..."
    
    mkdir -p "$PROJECT_DIR/docker/nginx/ssl"
    
    # Generate self-signed certificates for development
    openssl req -x509 -newkey rsa:4096 -keyout "$PROJECT_DIR/docker/nginx/ssl/tls.key" -out "$PROJECT_DIR/docker/nginx/ssl/tls.crt" -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    log_success "SSL certificates generated"
}

# Function to setup systemd services
setup_systemd_services() {
    log_message "Setting up systemd services..."
    
    # Create systemd service file
    cat > /tmp/autonomous-agent.service <<EOF
[Unit]
Description=Autonomous Agent System
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=agent
Group=agent
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/autonomous-agent.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable autonomous-agent
    
    log_success "Systemd services setup completed"
}

# Function to setup log rotation
setup_log_rotation() {
    log_message "Setting up log rotation..."
    
    cat > /tmp/autonomous-agent <<EOF
/opt/autonomous-agent/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 agent agent
    postrotate
        systemctl reload autonomous-agent
    endscript
}
EOF
    
    sudo mv /tmp/autonomous-agent /etc/logrotate.d/
    
    log_success "Log rotation setup completed"
}

# Function to setup backup cron job
setup_backup_cron() {
    log_message "Setting up backup cron job..."
    
    # Add backup cron job
    (crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_DIR/scripts/backup-restore.sh backup") | crontab -
    
    log_success "Backup cron job setup completed"
}

# Function to run security scan
run_security_scan() {
    log_message "Running security scan..."
    
    "$PROJECT_DIR/scripts/security-scan.sh" all
    
    log_success "Security scan completed"
}

# Function to verify installation
verify_installation() {
    log_message "Verifying installation..."
    
    local errors=0
    
    # Check Python environment
    if [ ! -f "$PROJECT_DIR/venv/bin/python" ]; then
        log_error "Python virtual environment not found"
        errors=$((errors + 1))
    fi
    
    # Check database connection
    if ! PGPASSWORD=agent123 psql -h localhost -U agent -d autonomous_agent -c "SELECT 1;" >/dev/null 2>&1; then
        log_error "Database connection failed"
        errors=$((errors + 1))
    fi
    
    # Check Redis connection
    if ! redis-cli ping >/dev/null 2>&1; then
        log_error "Redis connection failed"
        errors=$((errors + 1))
    fi
    
    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker not running"
        errors=$((errors + 1))
    fi
    
    # Check secrets
    if [ ! -f "$PROJECT_DIR/secrets/jwt_secret.txt" ]; then
        log_error "Secrets not generated"
        errors=$((errors + 1))
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "Installation verification passed"
    else
        log_error "Installation verification failed with $errors errors"
        exit 1
    fi
}

# Function to display setup summary
show_setup_summary() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Setup Summary${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo "Project Directory: $PROJECT_DIR"
    echo "Python Environment: $PROJECT_DIR/venv"
    echo "Database: PostgreSQL (autonomous_agent)"
    echo "Cache: Redis"
    echo "Monitoring: Prometheus + Grafana"
    echo "SSL: Self-signed certificates"
    echo ""
    echo -e "${GREEN}Next Steps:${NC}"
    echo "1. Start the application:"
    echo "   cd $PROJECT_DIR"
    echo "   source venv/bin/activate"
    echo "   python src/main.py"
    echo ""
    echo "2. Or use Docker Compose:"
    echo "   docker-compose up -d"
    echo ""
    echo "3. Access the application:"
    echo "   - Main app: http://localhost:8000"
    echo "   - Monitoring: http://localhost:3000"
    echo "   - Metrics: http://localhost:9090"
    echo ""
    echo "4. Configure external services:"
    echo "   - Update secrets in $PROJECT_DIR/secrets/"
    echo "   - Configure Gmail API credentials"
    echo "   - Configure GitHub token"
    echo ""
    echo -e "${GREEN}Setup completed successfully!${NC}"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --full              Full installation (default)"
    echo "  --minimal           Minimal installation"
    echo "  --docker-only       Docker-only setup"
    echo "  --dev               Development setup"
    echo "  --prod              Production setup"
    echo "  --skip-deps         Skip system dependencies"
    echo "  --skip-db           Skip database setup"
    echo "  --skip-redis        Skip Redis setup"
    echo "  --skip-docker       Skip Docker setup"
    echo "  --skip-monitoring   Skip monitoring setup"
    echo "  --skip-ssl          Skip SSL setup"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full installation"
    echo "  $0 --minimal        # Minimal installation"
    echo "  $0 --dev            # Development setup"
    echo "  $0 --prod           # Production setup"
    echo "  $0 --docker-only    # Docker-only setup"
}

# Main installation function
main() {
    echo -e "${BLUE}Autonomous Agent System Setup${NC}"
    echo -e "==============================="
    echo ""
    
    # Parse command line arguments
    FULL_INSTALL=true
    MINIMAL_INSTALL=false
    DOCKER_ONLY=false
    DEV_MODE=false
    PROD_MODE=false
    SKIP_DEPS=false
    SKIP_DB=false
    SKIP_REDIS=false
    SKIP_DOCKER=false
    SKIP_MONITORING=false
    SKIP_SSL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                FULL_INSTALL=true
                shift
                ;;
            --minimal)
                MINIMAL_INSTALL=true
                FULL_INSTALL=false
                shift
                ;;
            --docker-only)
                DOCKER_ONLY=true
                FULL_INSTALL=false
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --prod)
                PROD_MODE=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-db)
                SKIP_DB=true
                shift
                ;;
            --skip-redis)
                SKIP_REDIS=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --skip-ssl)
                SKIP_SSL=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Docker-only setup
    if [ "$DOCKER_ONLY" = true ]; then
        log_message "Docker-only setup mode"
        if [ "$SKIP_DOCKER" = false ]; then
            setup_docker
        fi
        generate_secrets
        if [ "$SKIP_SSL" = false ]; then
            setup_ssl
        fi
        log_success "Docker-only setup completed"
        echo ""
        echo "To start the application:"
        echo "  docker-compose up -d"
        exit 0
    fi
    
    # Install system dependencies
    if [ "$SKIP_DEPS" = false ]; then
        install_system_dependencies
    fi
    
    # Setup Python environment
    setup_python_environment
    
    # Setup database
    if [ "$SKIP_DB" = false ]; then
        setup_database
    fi
    
    # Setup Redis
    if [ "$SKIP_REDIS" = false ]; then
        setup_redis
    fi
    
    # Setup Docker
    if [ "$SKIP_DOCKER" = false ]; then
        setup_docker
    fi
    
    # Setup monitoring (if not minimal)
    if [ "$MINIMAL_INSTALL" = false ] && [ "$SKIP_MONITORING" = false ]; then
        setup_monitoring
    fi
    
    # Generate secrets
    generate_secrets
    
    # Setup SSL certificates
    if [ "$SKIP_SSL" = false ]; then
        setup_ssl
    fi
    
    # Production-specific setup
    if [ "$PROD_MODE" = true ]; then
        setup_systemd_services
        setup_log_rotation
        setup_backup_cron
    fi
    
    # Run security scan (if not minimal)
    if [ "$MINIMAL_INSTALL" = false ] && [ "$DEV_MODE" = false ]; then
        run_security_scan
    fi
    
    # Verify installation
    verify_installation
    
    # Show setup summary
    show_setup_summary
}

# Run main function
main "$@"