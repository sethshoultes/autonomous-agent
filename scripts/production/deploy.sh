#!/bin/bash

# ============================================================================
# Production Deployment Script
# ============================================================================
# This script handles the complete production deployment of the autonomous agent
# system with proper validation, rollback capabilities, and monitoring.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-rolling}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Deployment state tracking
DEPLOYMENT_STATE_FILE="/tmp/autonomous-agent-deployment.state"
DEPLOYMENT_BACKUP_DIR="/opt/autonomous-agent/backups/$(date +%Y%m%d_%H%M%S)"

# Cleanup function
cleanup() {
    local exit_code=$?
    log_info "Cleaning up deployment artifacts..."
    
    # Remove temporary files
    [ -f "$DEPLOYMENT_STATE_FILE" ] && rm -f "$DEPLOYMENT_STATE_FILE"
    
    # If deployment failed, attempt rollback
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        if [ -d "$DEPLOYMENT_BACKUP_DIR" ]; then
            log_info "Attempting automatic rollback..."
            perform_rollback
        fi
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Pre-deployment validation
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "kubectl" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            return 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running or not accessible"
        return 1
    fi
    
    # Check environment variables
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '$var' not set"
            return 1
        fi
    done
    
    # Check disk space (minimum 10GB)
    local available_space=$(df / | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        log_error "Insufficient disk space. At least 10GB required"
        return 1
    fi
    
    # Check memory (minimum 8GB)
    local available_memory=$(free -k | grep '^Mem:' | awk '{print $7}')
    if [ "$available_memory" -lt 8388608 ]; then  # 8GB in KB
        log_error "Insufficient memory. At least 8GB required"
        return 1
    fi
    
    log_success "Environment validation completed"
    return 0
}

# Security validation
validate_security() {
    log_info "Validating security configuration..."
    
    # Check secrets directory permissions
    if [ -d "${PROJECT_ROOT}/secrets" ]; then
        local secrets_perms=$(stat -c "%a" "${PROJECT_ROOT}/secrets")
        if [ "$secrets_perms" != "700" ]; then
            log_error "Secrets directory permissions incorrect: $secrets_perms (should be 700)"
            return 1
        fi
    fi
    
    # Check for required secrets
    local required_secrets=("postgres_password.txt" "redis_password.txt" "jwt_secret.txt")
    for secret in "${required_secrets[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/secrets/$secret" ]; then
            log_error "Required secret file not found: $secret"
            return 1
        fi
    done
    
    # Run security scan
    log_info "Running security scan..."
    if ! "${PROJECT_ROOT}/scripts/security-scan.sh" --quick; then
        log_error "Security scan failed"
        return 1
    fi
    
    log_success "Security validation completed"
    return 0
}

# Create deployment backup
create_deployment_backup() {
    log_info "Creating deployment backup..."
    
    # Create backup directory
    mkdir -p "$DEPLOYMENT_BACKUP_DIR"
    
    # Backup current configuration
    if [ -d "${PROJECT_ROOT}/config" ]; then
        cp -r "${PROJECT_ROOT}/config" "$DEPLOYMENT_BACKUP_DIR/"
    fi
    
    # Backup current containers state
    docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" config > "$DEPLOYMENT_BACKUP_DIR/docker-compose.backup.yml"
    
    # Backup database
    if docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" ps | grep -q postgres; then
        log_info "Creating database backup..."
        docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" exec -T postgres pg_dump -U agent autonomous_agent > "$DEPLOYMENT_BACKUP_DIR/database.sql"
    fi
    
    # Backup volumes
    log_info "Creating volume backups..."
    docker run --rm \
        -v autonomous-agent_postgres_data:/source:ro \
        -v "$DEPLOYMENT_BACKUP_DIR":/backup \
        alpine tar czf /backup/postgres_data.tar.gz -C /source .
    
    docker run --rm \
        -v autonomous-agent_redis_data:/source:ro \
        -v "$DEPLOYMENT_BACKUP_DIR":/backup \
        alpine tar czf /backup/redis_data.tar.gz -C /source .
    
    # Save deployment state
    echo "DEPLOYMENT_BACKUP_DIR=$DEPLOYMENT_BACKUP_DIR" > "$DEPLOYMENT_STATE_FILE"
    echo "DEPLOYMENT_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DEPLOYMENT_STATE_FILE"
    echo "DEPLOYMENT_VERSION=$(git rev-parse HEAD)" >> "$DEPLOYMENT_STATE_FILE"
    
    log_success "Deployment backup created at $DEPLOYMENT_BACKUP_DIR"
}

# Build and tag images
build_images() {
    log_info "Building production images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$(git describe --tags --always)" \
        -t autonomous-agent:latest \
        -t autonomous-agent:$(git rev-parse --short HEAD) \
        .
    
    # Build security scanner image
    docker build \
        --target security-scan \
        -t autonomous-agent-security:latest \
        .
    
    log_success "Images built successfully"
}

# Health check function
health_check() {
    local service_name="$1"
    local health_url="$2"
    local timeout="${3:-60}"
    
    log_info "Performing health check for $service_name..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [ $(date +%s) -lt $end_time ]; do
        if curl -f -s "$health_url" > /dev/null; then
            log_success "$service_name health check passed"
            return 0
        fi
        sleep 5
    done
    
    log_error "$service_name health check failed after ${timeout}s"
    return 1
}

# Rolling deployment
perform_rolling_deployment() {
    log_info "Starting rolling deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy infrastructure services first
    log_info "Deploying infrastructure services..."
    docker-compose -f docker-compose.prod.yml up -d postgres redis ollama nginx
    
    # Wait for infrastructure to be ready
    sleep 30
    
    # Health check infrastructure
    if ! health_check "PostgreSQL" "postgres://agent:${POSTGRES_PASSWORD}@localhost:5432/autonomous_agent"; then
        log_error "PostgreSQL health check failed"
        return 1
    fi
    
    if ! health_check "Redis" "redis://localhost:6379"; then
        log_error "Redis health check failed"
        return 1
    fi
    
    # Deploy application with rolling update
    log_info "Deploying application services..."
    docker-compose -f docker-compose.prod.yml up -d app
    
    # Wait for application to start
    sleep 60
    
    # Health check application
    if ! health_check "Application" "http://localhost:8000/health" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Application health check failed"
        return 1
    fi
    
    # Deploy monitoring services
    log_info "Deploying monitoring services..."
    docker-compose -f docker-compose.prod.yml up -d prometheus grafana elasticsearch kibana
    
    # Wait for monitoring to be ready
    sleep 30
    
    log_success "Rolling deployment completed successfully"
}

# Blue-green deployment
perform_blue_green_deployment() {
    log_info "Starting blue-green deployment..."
    
    # Create green environment
    log_info "Creating green environment..."
    
    # Modify docker-compose for green deployment
    sed 's/autonomous-agent-/autonomous-agent-green-/g' docker-compose.prod.yml > docker-compose.green.yml
    sed -i 's/8000:8000/8001:8000/g' docker-compose.green.yml
    
    # Deploy green environment
    docker-compose -f docker-compose.green.yml up -d
    
    # Wait for green environment to be ready
    sleep 60
    
    # Health check green environment
    if ! health_check "Green Application" "http://localhost:8001/health" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Green environment health check failed"
        docker-compose -f docker-compose.green.yml down
        rm -f docker-compose.green.yml
        return 1
    fi
    
    # Switch traffic to green
    log_info "Switching traffic to green environment..."
    
    # Update nginx configuration to point to green
    # This would typically involve updating load balancer configuration
    
    # Stop blue environment
    docker-compose -f docker-compose.prod.yml down
    
    # Rename green to blue
    docker-compose -f docker-compose.green.yml down
    mv docker-compose.green.yml docker-compose.prod.yml
    docker-compose -f docker-compose.prod.yml up -d
    
    log_success "Blue-green deployment completed successfully"
}

# Canary deployment
perform_canary_deployment() {
    log_info "Starting canary deployment..."
    
    # Deploy canary version alongside current version
    log_info "Deploying canary version..."
    
    # Create canary compose file
    sed 's/autonomous-agent-/autonomous-agent-canary-/g' docker-compose.prod.yml > docker-compose.canary.yml
    sed -i 's/8000:8000/8002:8000/g' docker-compose.canary.yml
    
    # Deploy canary
    docker-compose -f docker-compose.canary.yml up -d app
    
    # Wait for canary to be ready
    sleep 60
    
    # Health check canary
    if ! health_check "Canary Application" "http://localhost:8002/health" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Canary health check failed"
        docker-compose -f docker-compose.canary.yml down
        rm -f docker-compose.canary.yml
        return 1
    fi
    
    # Monitor canary for issues (simplified - would integrate with monitoring)
    log_info "Monitoring canary deployment..."
    sleep 300  # 5 minutes monitoring
    
    # If canary is healthy, gradually increase traffic
    log_info "Promoting canary to full deployment..."
    
    # Stop original version
    docker-compose -f docker-compose.prod.yml stop app
    
    # Scale up canary
    docker-compose -f docker-compose.canary.yml up -d --scale app=2
    
    # Final health check
    if ! health_check "Canary Application" "http://localhost:8002/health" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Canary promotion failed"
        return 1
    fi
    
    # Clean up
    docker-compose -f docker-compose.prod.yml down
    mv docker-compose.canary.yml docker-compose.prod.yml
    
    log_success "Canary deployment completed successfully"
}

# Rollback function
perform_rollback() {
    log_info "Starting rollback procedure..."
    
    if [ ! -f "$DEPLOYMENT_STATE_FILE" ]; then
        log_error "No deployment state file found. Cannot perform rollback."
        return 1
    fi
    
    # Load deployment state
    source "$DEPLOYMENT_STATE_FILE"
    
    if [ ! -d "$DEPLOYMENT_BACKUP_DIR" ]; then
        log_error "Backup directory not found: $DEPLOYMENT_BACKUP_DIR"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose -f docker-compose.prod.yml down
    
    # Restore configuration
    if [ -d "$DEPLOYMENT_BACKUP_DIR/config" ]; then
        log_info "Restoring configuration..."
        rm -rf config
        cp -r "$DEPLOYMENT_BACKUP_DIR/config" .
    fi
    
    # Restore database
    if [ -f "$DEPLOYMENT_BACKUP_DIR/database.sql" ]; then
        log_info "Restoring database..."
        docker-compose -f docker-compose.prod.yml up -d postgres
        sleep 30
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -d autonomous_agent < "$DEPLOYMENT_BACKUP_DIR/database.sql"
    fi
    
    # Restore volumes
    log_info "Restoring volumes..."
    docker run --rm \
        -v autonomous-agent_postgres_data:/target \
        -v "$DEPLOYMENT_BACKUP_DIR":/backup:ro \
        alpine tar xzf /backup/postgres_data.tar.gz -C /target
    
    docker run --rm \
        -v autonomous-agent_redis_data:/target \
        -v "$DEPLOYMENT_BACKUP_DIR":/backup:ro \
        alpine tar xzf /backup/redis_data.tar.gz -C /target
    
    # Start services with backup configuration
    log_info "Starting services with backup configuration..."
    docker-compose -f "$DEPLOYMENT_BACKUP_DIR/docker-compose.backup.yml" up -d
    
    # Wait for services to be ready
    sleep 60
    
    # Health check
    if ! health_check "Rollback Application" "http://localhost:8000/health" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Rollback health check failed"
        return 1
    fi
    
    log_success "Rollback completed successfully"
}

# Post-deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check all services are running
    local services=("app" "postgres" "redis" "ollama" "nginx")
    for service in "${services[@]}"; do
        if ! docker-compose -f docker-compose.prod.yml ps "$service" | grep -q "Up"; then
            log_error "Service $service is not running"
            return 1
        fi
    done
    
    # Run comprehensive health checks
    health_check "Application" "http://localhost:8000/health" 60
    health_check "API Status" "http://localhost:8000/api/v1/status" 30
    
    # Test key functionality
    log_info "Testing key functionality..."
    
    # Test Gmail agent
    if curl -s -f "http://localhost:8000/api/v1/agents/gmail/health" > /dev/null; then
        log_success "Gmail agent is healthy"
    else
        log_warning "Gmail agent health check failed"
    fi
    
    # Test Research agent
    if curl -s -f "http://localhost:8000/api/v1/agents/research/health" > /dev/null; then
        log_success "Research agent is healthy"
    else
        log_warning "Research agent health check failed"
    fi
    
    # Test Code agent
    if curl -s -f "http://localhost:8000/api/v1/agents/code/health" > /dev/null; then
        log_success "Code agent is healthy"
    else
        log_warning "Code agent health check failed"
    fi
    
    # Test Intelligence Engine
    if curl -s -f "http://localhost:8000/api/v1/intelligence/health" > /dev/null; then
        log_success "Intelligence Engine is healthy"
    else
        log_warning "Intelligence Engine health check failed"
    fi
    
    log_success "Deployment validation completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    local backup_base_dir="/opt/autonomous-agent/backups"
    if [ -d "$backup_base_dir" ]; then
        find "$backup_base_dir" -type d -name "*_*" -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} \;
        log_success "Old backups cleaned up"
    fi
}

# Main deployment function
main() {
    log_info "Starting production deployment for autonomous-agent"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $DEPLOYMENT_ENV"
    
    # Pre-deployment validation
    validate_environment
    validate_security
    
    # Create backup before deployment
    create_deployment_backup
    
    # Build images
    build_images
    
    # Perform deployment based on type
    case "$DEPLOYMENT_TYPE" in
        "rolling")
            perform_rolling_deployment
            ;;
        "blue-green")
            perform_blue_green_deployment
            ;;
        "canary")
            perform_canary_deployment
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Post-deployment validation
    validate_deployment
    
    # Cleanup old backups
    cleanup_old_backups
    
    log_success "Production deployment completed successfully!"
    log_info "Deployment backup saved at: $DEPLOYMENT_BACKUP_DIR"
    log_info "Application available at: http://localhost:8000"
    log_info "Monitoring available at: http://localhost:3000"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi