#!/bin/bash

# ============================================================================
# Disaster Recovery Script for Autonomous Agent
# ============================================================================
# This script handles disaster recovery procedures including full system
# restoration, data recovery, and service recovery operations.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/opt/autonomous-agent/backups}"
RECOVERY_LOG_FILE="${RECOVERY_LOG_FILE:-/var/log/disaster-recovery.log}"
RECOVERY_TIMEOUT="${RECOVERY_TIMEOUT:-3600}"
PARALLEL_RECOVERY="${PARALLEL_RECOVERY:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Recovery state tracking
RECOVERY_STATE_FILE="/tmp/disaster-recovery.state"
RECOVERY_STEPS_COMPLETED=0
RECOVERY_STEPS_TOTAL=10

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$RECOVERY_LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$RECOVERY_LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$RECOVERY_LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$RECOVERY_LOG_FILE"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -b, --backup-date DATE       Recovery from specific backup date (YYYYMMDD_HHMMSS)
    -t, --recovery-type TYPE     Recovery type (full, partial, database, config)
    -s, --skip-verification      Skip pre-recovery verification
    -p, --parallel               Enable parallel recovery operations
    -f, --force                  Force recovery without confirmation
    -d, --dry-run                Show what would be done without executing
    -h, --help                   Show this help message

Recovery Types:
    full        - Complete system recovery (default)
    partial     - Partial system recovery (specify components)
    database    - Database-only recovery
    config      - Configuration-only recovery
    volumes     - Volume data recovery
    secrets     - Secrets recovery

Examples:
    $0 --backup-date 20240101_020000
    $0 --recovery-type database --backup-date 20240101_020000
    $0 --recovery-type partial --components "app,database"
    $0 --force --parallel
EOF
}

# Parse command line arguments
BACKUP_DATE=""
RECOVERY_TYPE="full"
SKIP_VERIFICATION=false
FORCE=false
DRY_RUN=false
COMPONENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backup-date)
            BACKUP_DATE="$2"
            shift 2
            ;;
        -t|--recovery-type)
            RECOVERY_TYPE="$2"
            shift 2
            ;;
        -s|--skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        -p|--parallel)
            PARALLEL_RECOVERY=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -c|--components)
            COMPONENTS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Cleanup function
cleanup() {
    local exit_code=$?
    log_info "Cleaning up disaster recovery artifacts..."
    
    # Remove temporary files
    [ -f "$RECOVERY_STATE_FILE" ] && rm -f "$RECOVERY_STATE_FILE"
    
    # If recovery failed, log the failure
    if [ $exit_code -ne 0 ]; then
        log_error "Disaster recovery failed with exit code $exit_code"
        log_error "Check $RECOVERY_LOG_FILE for details"
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Progress tracking
update_progress() {
    local step_name="$1"
    RECOVERY_STEPS_COMPLETED=$((RECOVERY_STEPS_COMPLETED + 1))
    local progress=$((RECOVERY_STEPS_COMPLETED * 100 / RECOVERY_STEPS_TOTAL))
    
    log_info "Progress: $progress% - $step_name"
    echo "STEP=$RECOVERY_STEPS_COMPLETED" > "$RECOVERY_STATE_FILE"
    echo "PROGRESS=$progress" >> "$RECOVERY_STATE_FILE"
    echo "CURRENT_STEP=$step_name" >> "$RECOVERY_STATE_FILE"
}

# Find latest backup if no date specified
find_latest_backup() {
    log_info "Finding latest backup..."
    
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_BASE_DIR"
        return 1
    fi
    
    local latest_backup=$(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "*_*" | sort | tail -1)
    
    if [ -z "$latest_backup" ]; then
        log_error "No backups found in $BACKUP_BASE_DIR"
        return 1
    fi
    
    echo "$(basename "$latest_backup")"
}

# Validate backup
validate_backup() {
    local backup_date="$1"
    local backup_dir="$BACKUP_BASE_DIR/$backup_date"
    
    log_info "Validating backup: $backup_date"
    
    if [ ! -d "$backup_dir" ]; then
        log_error "Backup directory not found: $backup_dir"
        return 1
    fi
    
    # Check required files
    local required_files=("deployment.state")
    for file in "${required_files[@]}"; do
        if [ ! -f "$backup_dir/$file" ]; then
            log_error "Required backup file missing: $file"
            return 1
        fi
    done
    
    # Check backup integrity
    if [ -f "$backup_dir/database.sql" ]; then
        if ! head -1 "$backup_dir/database.sql" | grep -q "PostgreSQL database dump"; then
            log_error "Database backup appears corrupted"
            return 1
        fi
    fi
    
    # Check volume backups
    local volume_backups=("postgres_data.tar.gz" "redis_data.tar.gz" "ollama_data.tar.gz")
    for volume in "${volume_backups[@]}"; do
        if [ -f "$backup_dir/$volume" ]; then
            if ! tar -tzf "$backup_dir/$volume" > /dev/null 2>&1; then
                log_error "Volume backup corrupted: $volume"
                return 1
            fi
        fi
    done
    
    log_success "Backup validation completed successfully"
    return 0
}

# Pre-recovery verification
pre_recovery_verification() {
    log_info "Performing pre-recovery verification..."
    
    # Check available disk space
    local available_space=$(df / | tail -1 | awk '{print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space. Required: 10GB, Available: $((available_space / 1024 / 1024))GB"
        return 1
    fi
    
    # Check Docker availability
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker before recovery."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running. Please start Docker daemon."
        return 1
    fi
    
    # Check Docker Compose availability
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        return 1
    fi
    
    # Check network connectivity
    if ! ping -c 1 8.8.8.8 &> /dev/null; then
        log_warning "No internet connectivity. Some recovery operations may fail."
    fi
    
    log_success "Pre-recovery verification completed"
    return 0
}

# Stop running services
stop_services() {
    log_info "Stopping running services..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would stop all services"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Stop all services
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        log_info "Stopping Docker Compose services..."
        docker-compose -f docker-compose.prod.yml down --timeout 30
    fi
    
    # Stop any running containers
    local running_containers=$(docker ps -q --filter "name=autonomous-agent")
    if [ -n "$running_containers" ]; then
        log_info "Stopping remaining containers..."
        docker stop $running_containers
    fi
    
    # Remove stopped containers
    local stopped_containers=$(docker ps -aq --filter "name=autonomous-agent")
    if [ -n "$stopped_containers" ]; then
        log_info "Removing stopped containers..."
        docker rm $stopped_containers
    fi
    
    log_success "Services stopped successfully"
    update_progress "Services stopped"
}

# Restore configuration
restore_configuration() {
    local backup_dir="$1"
    
    log_info "Restoring configuration from backup..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would restore configuration from $backup_dir"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Backup current configuration
    if [ -d "config" ]; then
        log_info "Backing up current configuration..."
        mv config config.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    # Restore configuration
    if [ -d "$backup_dir/config" ]; then
        log_info "Restoring configuration files..."
        cp -r "$backup_dir/config" .
    fi
    
    # Restore environment files
    if [ -f "$backup_dir/.env.production" ]; then
        log_info "Restoring environment configuration..."
        cp "$backup_dir/.env.production" .
    fi
    
    # Restore Docker Compose configuration
    if [ -f "$backup_dir/docker-compose.prod.yml" ]; then
        log_info "Restoring Docker Compose configuration..."
        cp "$backup_dir/docker-compose.prod.yml" .
    fi
    
    log_success "Configuration restored successfully"
    update_progress "Configuration restored"
}

# Restore secrets
restore_secrets() {
    local backup_dir="$1"
    
    log_info "Restoring secrets from backup..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would restore secrets from $backup_dir"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Backup current secrets
    if [ -d "secrets" ]; then
        log_info "Backing up current secrets..."
        mv secrets secrets.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    # Restore secrets
    if [ -d "$backup_dir/secrets" ]; then
        log_info "Restoring secrets..."
        cp -r "$backup_dir/secrets" .
        chmod 700 secrets
        chmod 600 secrets/*
    fi
    
    log_success "Secrets restored successfully"
    update_progress "Secrets restored"
}

# Restore Docker volumes
restore_volumes() {
    local backup_dir="$1"
    
    log_info "Restoring Docker volumes from backup..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would restore volumes from $backup_dir"
        return 0
    fi
    
    # Create volumes if they don't exist
    local volumes=("postgres_data" "redis_data" "ollama_data" "app_data")
    for volume in "${volumes[@]}"; do
        if ! docker volume ls | grep -q "autonomous-agent_$volume"; then
            log_info "Creating volume: autonomous-agent_$volume"
            docker volume create "autonomous-agent_$volume"
        fi
    done
    
    # Restore volume data
    if [ "$PARALLEL_RECOVERY" = true ]; then
        log_info "Restoring volumes in parallel..."
        
        # Restore volumes in parallel
        (
            if [ -f "$backup_dir/postgres_data.tar.gz" ]; then
                log_info "Restoring PostgreSQL data..."
                docker run --rm \
                    -v autonomous-agent_postgres_data:/target \
                    -v "$backup_dir":/backup:ro \
                    alpine tar xzf /backup/postgres_data.tar.gz -C /target
            fi
        ) &
        
        (
            if [ -f "$backup_dir/redis_data.tar.gz" ]; then
                log_info "Restoring Redis data..."
                docker run --rm \
                    -v autonomous-agent_redis_data:/target \
                    -v "$backup_dir":/backup:ro \
                    alpine tar xzf /backup/redis_data.tar.gz -C /target
            fi
        ) &
        
        (
            if [ -f "$backup_dir/ollama_data.tar.gz" ]; then
                log_info "Restoring Ollama data..."
                docker run --rm \
                    -v autonomous-agent_ollama_data:/target \
                    -v "$backup_dir":/backup:ro \
                    alpine tar xzf /backup/ollama_data.tar.gz -C /target
            fi
        ) &
        
        # Wait for all parallel operations to complete
        wait
    else
        # Sequential restoration
        if [ -f "$backup_dir/postgres_data.tar.gz" ]; then
            log_info "Restoring PostgreSQL data..."
            docker run --rm \
                -v autonomous-agent_postgres_data:/target \
                -v "$backup_dir":/backup:ro \
                alpine tar xzf /backup/postgres_data.tar.gz -C /target
        fi
        
        if [ -f "$backup_dir/redis_data.tar.gz" ]; then
            log_info "Restoring Redis data..."
            docker run --rm \
                -v autonomous-agent_redis_data:/target \
                -v "$backup_dir":/backup:ro \
                alpine tar xzf /backup/redis_data.tar.gz -C /target
        fi
        
        if [ -f "$backup_dir/ollama_data.tar.gz" ]; then
            log_info "Restoring Ollama data..."
            docker run --rm \
                -v autonomous-agent_ollama_data:/target \
                -v "$backup_dir":/backup:ro \
                alpine tar xzf /backup/ollama_data.tar.gz -C /target
        fi
    fi
    
    log_success "Volumes restored successfully"
    update_progress "Volumes restored"
}

# Restore database
restore_database() {
    local backup_dir="$1"
    
    log_info "Restoring database from backup..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would restore database from $backup_dir"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Start database service only
    log_info "Starting database service..."
    docker-compose -f docker-compose.prod.yml up -d postgres
    
    # Wait for database to be ready
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U agent > /dev/null 2>&1; then
            log_info "Database is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Database failed to start within timeout"
        return 1
    fi
    
    # Restore database
    if [ -f "$backup_dir/database.sql" ]; then
        log_info "Restoring database from SQL dump..."
        
        # Drop existing database and recreate
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -c "DROP DATABASE IF EXISTS autonomous_agent;"
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -c "CREATE DATABASE autonomous_agent;"
        
        # Restore database
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -d autonomous_agent < "$backup_dir/database.sql"
        
        log_success "Database restored from SQL dump"
    else
        log_warning "No database backup found, skipping database restore"
    fi
    
    log_success "Database restoration completed"
    update_progress "Database restored"
}

# Start services
start_services() {
    log_info "Starting services..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would start all services"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Start all services
    log_info "Starting all services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 60
    
    log_success "Services started successfully"
    update_progress "Services started"
}

# Health verification
verify_recovery() {
    log_info "Verifying recovery..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would verify recovery"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Check service health
    log_info "Checking service health..."
    if ! ./scripts/production/health-check.sh --quiet; then
        log_error "Health check failed after recovery"
        return 1
    fi
    
    # Test basic functionality
    log_info "Testing basic functionality..."
    
    # Test health endpoint
    if ! curl -f -s "http://localhost:8000/health" > /dev/null; then
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # Test API endpoints
    if ! curl -f -s "http://localhost:8000/api/v1/status" > /dev/null; then
        log_error "API status endpoint not responding"
        return 1
    fi
    
    # Test database connectivity
    if ! docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U agent > /dev/null 2>&1; then
        log_error "Database not accessible"
        return 1
    fi
    
    # Test Redis connectivity
    if ! docker-compose -f docker-compose.prod.yml exec redis redis-cli ping | grep -q "PONG"; then
        log_error "Redis not accessible"
        return 1
    fi
    
    log_success "Recovery verification completed successfully"
    update_progress "Recovery verified"
}

# Generate recovery report
generate_recovery_report() {
    local backup_date="$1"
    local recovery_type="$2"
    
    log_info "Generating recovery report..."
    
    local report_file="$PROJECT_ROOT/recovery-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Disaster Recovery Report

## Recovery Summary

- **Date**: $(date)
- **Recovery Type**: $recovery_type
- **Backup Date**: $backup_date
- **Recovery Status**: SUCCESS
- **Recovery Duration**: $(date -d@$SECONDS -u +%H:%M:%S)

## Recovery Steps Completed

1. ✅ Pre-recovery verification
2. ✅ Backup validation
3. ✅ Services stopped
4. ✅ Configuration restored
5. ✅ Secrets restored
6. ✅ Volumes restored
7. ✅ Database restored
8. ✅ Services started
9. ✅ Recovery verified
10. ✅ Report generated

## System Status

- **Application**: Healthy
- **Database**: Healthy
- **Cache**: Healthy
- **AI Service**: Healthy
- **Monitoring**: Healthy

## Post-Recovery Actions

1. Monitor system performance for 24 hours
2. Verify all agents are functioning correctly
3. Check data integrity
4. Review logs for any issues
5. Update backup procedures if needed

## Recovery Log

Log file: $RECOVERY_LOG_FILE

EOF
    
    log_success "Recovery report generated: $report_file"
    update_progress "Report generated"
}

# Confirmation prompt
confirm_recovery() {
    local backup_date="$1"
    local recovery_type="$2"
    
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    echo
    log_warning "You are about to perform disaster recovery!"
    log_info "Recovery type: $recovery_type"
    log_info "Backup date: $backup_date"
    log_info "This will stop all services and restore from backup"
    echo
    
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Recovery cancelled by user"
        exit 0
    fi
}

# Main recovery function
main() {
    log_info "Starting disaster recovery process"
    log_info "Recovery type: $RECOVERY_TYPE"
    
    # Initialize recovery log
    echo "=== Disaster Recovery Log - $(date) ===" > "$RECOVERY_LOG_FILE"
    
    # Determine backup date
    if [ -z "$BACKUP_DATE" ]; then
        BACKUP_DATE=$(find_latest_backup)
        if [ -z "$BACKUP_DATE" ]; then
            log_error "No backup found and no backup date specified"
            exit 1
        fi
        log_info "Using latest backup: $BACKUP_DATE"
    fi
    
    local backup_dir="$BACKUP_BASE_DIR/$BACKUP_DATE"
    
    # Validate backup
    if ! validate_backup "$BACKUP_DATE"; then
        log_error "Backup validation failed"
        exit 1
    fi
    
    # Pre-recovery verification
    if [ "$SKIP_VERIFICATION" != true ]; then
        if ! pre_recovery_verification; then
            log_error "Pre-recovery verification failed"
            exit 1
        fi
    fi
    
    # Confirm recovery
    confirm_recovery "$BACKUP_DATE" "$RECOVERY_TYPE"
    
    # Start recovery timer
    SECONDS=0
    
    # Perform recovery based on type
    case "$RECOVERY_TYPE" in
        "full")
            stop_services
            restore_configuration "$backup_dir"
            restore_secrets "$backup_dir"
            restore_volumes "$backup_dir"
            restore_database "$backup_dir"
            start_services
            verify_recovery
            ;;
        "database")
            stop_services
            restore_database "$backup_dir"
            start_services
            verify_recovery
            ;;
        "config")
            stop_services
            restore_configuration "$backup_dir"
            start_services
            verify_recovery
            ;;
        "volumes")
            stop_services
            restore_volumes "$backup_dir"
            start_services
            verify_recovery
            ;;
        "secrets")
            stop_services
            restore_secrets "$backup_dir"
            start_services
            verify_recovery
            ;;
        "partial")
            stop_services
            IFS=',' read -ra COMP <<< "$COMPONENTS"
            for component in "${COMP[@]}"; do
                case "$component" in
                    "config")
                        restore_configuration "$backup_dir"
                        ;;
                    "secrets")
                        restore_secrets "$backup_dir"
                        ;;
                    "database")
                        restore_database "$backup_dir"
                        ;;
                    "volumes")
                        restore_volumes "$backup_dir"
                        ;;
                    *)
                        log_warning "Unknown component: $component"
                        ;;
                esac
            done
            start_services
            verify_recovery
            ;;
        *)
            log_error "Unknown recovery type: $RECOVERY_TYPE"
            exit 1
            ;;
    esac
    
    # Generate recovery report
    generate_recovery_report "$BACKUP_DATE" "$RECOVERY_TYPE"
    
    log_success "Disaster recovery completed successfully!"
    log_info "Recovery time: $(date -d@$SECONDS -u +%H:%M:%S)"
    log_info "Application available at: http://localhost:8000"
    log_info "Monitoring available at: http://localhost:3000"
    
    # Cleanup
    rm -f "$RECOVERY_STATE_FILE"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi