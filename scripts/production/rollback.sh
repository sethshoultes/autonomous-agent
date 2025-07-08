#!/bin/bash

# ============================================================================
# Production Rollback Script
# ============================================================================
# This script handles rollback operations for the autonomous agent system
# with comprehensive validation and recovery capabilities.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_BASE_DIR="/opt/autonomous-agent/backups"
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

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -b, --backup-dir DIR     Specific backup directory to rollback to
    -l, --list-backups       List available backups
    -t, --target-version VER Rollback to specific version
    -f, --force             Force rollback without confirmation
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

Examples:
    $0 --list-backups
    $0 --backup-dir /opt/autonomous-agent/backups/20240101_120000
    $0 --target-version v1.2.3
    $0 --force
EOF
}

# Parse command line arguments
BACKUP_DIR=""
LIST_BACKUPS=false
TARGET_VERSION=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -l|--list-backups)
            LIST_BACKUPS=true
            shift
            ;;
        -t|--target-version)
            TARGET_VERSION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
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

# List available backups
list_backups() {
    log_info "Available backups:"
    
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        log_warning "No backup directory found at $BACKUP_BASE_DIR"
        return 1
    fi
    
    local backups=()
    while IFS= read -r -d '' backup; do
        backups+=("$backup")
    done < <(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "*_*" -print0 | sort -z)
    
    if [ ${#backups[@]} -eq 0 ]; then
        log_warning "No backups found"
        return 1
    fi
    
    echo
    printf "%-20s %-20s %-15s %-30s\n" "Backup ID" "Date" "Version" "Path"
    printf "%-20s %-20s %-15s %-30s\n" "----------" "----" "-------" "----"
    
    for backup in "${backups[@]}"; do
        local backup_name=$(basename "$backup")
        local backup_date=$(echo "$backup_name" | sed 's/_/ /')
        local version="unknown"
        
        if [ -f "$backup/deployment.state" ]; then
            version=$(grep "DEPLOYMENT_VERSION=" "$backup/deployment.state" | cut -d'=' -f2)
        fi
        
        printf "%-20s %-20s %-15s %-30s\n" "$backup_name" "$backup_date" "$version" "$backup"
    done
    
    echo
}

# Find backup by version
find_backup_by_version() {
    local target_version="$1"
    
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_BASE_DIR"
        return 1
    fi
    
    local backups=()
    while IFS= read -r -d '' backup; do
        backups+=("$backup")
    done < <(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "*_*" -print0)
    
    for backup in "${backups[@]}"; do
        if [ -f "$backup/deployment.state" ]; then
            local version=$(grep "DEPLOYMENT_VERSION=" "$backup/deployment.state" | cut -d'=' -f2)
            if [ "$version" = "$target_version" ]; then
                echo "$backup"
                return 0
            fi
        fi
    done
    
    log_error "No backup found for version: $target_version"
    return 1
}

# Get latest backup
get_latest_backup() {
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_BASE_DIR"
        return 1
    fi
    
    local latest_backup=$(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "*_*" | sort | tail -1)
    
    if [ -z "$latest_backup" ]; then
        log_error "No backups found"
        return 1
    fi
    
    echo "$latest_backup"
}

# Validate backup directory
validate_backup() {
    local backup_dir="$1"
    
    log_info "Validating backup directory: $backup_dir"
    
    if [ ! -d "$backup_dir" ]; then
        log_error "Backup directory does not exist: $backup_dir"
        return 1
    fi
    
    # Check required files
    local required_files=("docker-compose.backup.yml" "deployment.state")
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
    local volume_backups=("postgres_data.tar.gz" "redis_data.tar.gz")
    for volume in "${volume_backups[@]}"; do
        if [ -f "$backup_dir/$volume" ]; then
            if ! tar -tzf "$backup_dir/$volume" > /dev/null 2>&1; then
                log_error "Volume backup corrupted: $volume"
                return 1
            fi
        fi
    done
    
    log_success "Backup validation completed"
    return 0
}

# Create pre-rollback backup
create_pre_rollback_backup() {
    log_info "Creating pre-rollback backup..."
    
    local pre_rollback_dir="/opt/autonomous-agent/backups/pre-rollback-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$pre_rollback_dir"
    
    # Backup current state
    cd "$PROJECT_ROOT"
    
    # Backup configuration
    if [ -d "config" ]; then
        cp -r config "$pre_rollback_dir/"
    fi
    
    # Backup current compose file
    if [ -f "docker-compose.prod.yml" ]; then
        cp docker-compose.prod.yml "$pre_rollback_dir/"
    fi
    
    # Backup database
    if docker-compose -f docker-compose.prod.yml ps postgres | grep -q "Up"; then
        log_info "Backing up current database..."
        docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U agent autonomous_agent > "$pre_rollback_dir/database.sql"
    fi
    
    # Backup volumes
    log_info "Backing up current volumes..."
    docker run --rm \
        -v autonomous-agent_postgres_data:/source:ro \
        -v "$pre_rollback_dir":/backup \
        alpine tar czf /backup/postgres_data.tar.gz -C /source . || true
    
    docker run --rm \
        -v autonomous-agent_redis_data:/source:ro \
        -v "$pre_rollback_dir":/backup \
        alpine tar czf /backup/redis_data.tar.gz -C /source . || true
    
    # Save current state
    echo "PRE_ROLLBACK_BACKUP_DIR=$pre_rollback_dir" > "$pre_rollback_dir/deployment.state"
    echo "ROLLBACK_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$pre_rollback_dir/deployment.state"
    echo "CURRENT_VERSION=$(git rev-parse HEAD)" >> "$pre_rollback_dir/deployment.state"
    
    log_success "Pre-rollback backup created at: $pre_rollback_dir"
    echo "$pre_rollback_dir"
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
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log_success "$service_name health check passed"
            return 0
        fi
        sleep 5
    done
    
    log_error "$service_name health check failed after ${timeout}s"
    return 1
}

# Perform rollback
perform_rollback() {
    local backup_dir="$1"
    
    log_info "Starting rollback from backup: $backup_dir"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would rollback from $backup_dir"
        log_info "DRY RUN: Would stop current services"
        log_info "DRY RUN: Would restore configuration from backup"
        log_info "DRY RUN: Would restore database from backup"
        log_info "DRY RUN: Would restore volumes from backup"
        log_info "DRY RUN: Would start services with backup configuration"
        return 0
    fi
    
    # Create pre-rollback backup
    local pre_rollback_backup=$(create_pre_rollback_backup)
    
    cd "$PROJECT_ROOT"
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose -f docker-compose.prod.yml down || true
    
    # Remove current volumes (if requested)
    if [ "$FORCE" = true ]; then
        log_warning "Force mode: Removing current volumes..."
        docker volume rm autonomous-agent_postgres_data autonomous-agent_redis_data || true
    fi
    
    # Restore configuration
    if [ -d "$backup_dir/config" ]; then
        log_info "Restoring configuration..."
        rm -rf config
        cp -r "$backup_dir/config" . || true
    fi
    
    # Restore docker-compose file
    if [ -f "$backup_dir/docker-compose.backup.yml" ]; then
        log_info "Restoring docker-compose configuration..."
        cp "$backup_dir/docker-compose.backup.yml" docker-compose.prod.yml
    fi
    
    # Start infrastructure services
    log_info "Starting infrastructure services..."
    docker-compose -f docker-compose.prod.yml up -d postgres redis ollama
    
    # Wait for infrastructure to be ready
    sleep 30
    
    # Restore database
    if [ -f "$backup_dir/database.sql" ]; then
        log_info "Restoring database..."
        # Wait for postgres to be ready
        local max_attempts=30
        local attempt=0
        while [ $attempt -lt $max_attempts ]; do
            if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U agent > /dev/null 2>&1; then
                break
            fi
            attempt=$((attempt + 1))
            sleep 2
        done
        
        # Drop and recreate database
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -c "DROP DATABASE IF EXISTS autonomous_agent;"
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -c "CREATE DATABASE autonomous_agent;"
        
        # Restore database
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U agent -d autonomous_agent < "$backup_dir/database.sql"
    fi
    
    # Restore volumes
    log_info "Restoring volumes..."
    if [ -f "$backup_dir/postgres_data.tar.gz" ]; then
        docker run --rm \
            -v autonomous-agent_postgres_data:/target \
            -v "$backup_dir":/backup:ro \
            alpine tar xzf /backup/postgres_data.tar.gz -C /target || true
    fi
    
    if [ -f "$backup_dir/redis_data.tar.gz" ]; then
        docker run --rm \
            -v autonomous-agent_redis_data:/target \
            -v "$backup_dir":/backup:ro \
            alpine tar xzf /backup/redis_data.tar.gz -C /target || true
    fi
    
    # Start all services
    log_info "Starting all services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    sleep 60
    
    # Perform health checks
    log_info "Performing post-rollback health checks..."
    
    if ! health_check "Application" "http://localhost:8000/health" "$ROLLBACK_TIMEOUT"; then
        log_error "Rollback failed: Application health check failed"
        log_info "Attempting to restore from pre-rollback backup..."
        perform_rollback "$pre_rollback_backup"
        return 1
    fi
    
    # Test key functionality
    log_info "Testing key functionality..."
    
    local services=("gmail" "research" "code" "intelligence")
    for service in "${services[@]}"; do
        if curl -s -f "http://localhost:8000/api/v1/agents/$service/health" > /dev/null 2>&1; then
            log_success "$service agent is healthy"
        else
            log_warning "$service agent health check failed"
        fi
    done
    
    log_success "Rollback completed successfully!"
    log_info "Pre-rollback backup available at: $pre_rollback_backup"
    
    # Load backup metadata
    if [ -f "$backup_dir/deployment.state" ]; then
        local backup_version=$(grep "DEPLOYMENT_VERSION=" "$backup_dir/deployment.state" | cut -d'=' -f2)
        local backup_time=$(grep "DEPLOYMENT_TIME=" "$backup_dir/deployment.state" | cut -d'=' -f2)
        log_info "Rolled back to version: $backup_version"
        log_info "Original deployment time: $backup_time"
    fi
}

# Confirmation prompt
confirm_rollback() {
    local backup_dir="$1"
    
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    echo
    log_warning "You are about to rollback the autonomous agent system!"
    log_info "Rollback source: $backup_dir"
    
    if [ -f "$backup_dir/deployment.state" ]; then
        local backup_version=$(grep "DEPLOYMENT_VERSION=" "$backup_dir/deployment.state" | cut -d'=' -f2)
        local backup_time=$(grep "DEPLOYMENT_TIME=" "$backup_dir/deployment.state" | cut -d'=' -f2)
        log_info "Target version: $backup_version"
        log_info "Backup time: $backup_time"
    fi
    
    echo
    read -p "Are you sure you want to proceed with the rollback? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

# Main function
main() {
    log_info "Starting autonomous agent rollback procedure"
    
    # Handle list backups
    if [ "$LIST_BACKUPS" = true ]; then
        list_backups
        exit 0
    fi
    
    # Determine backup directory
    local backup_dir=""
    
    if [ -n "$BACKUP_DIR" ]; then
        backup_dir="$BACKUP_DIR"
    elif [ -n "$TARGET_VERSION" ]; then
        backup_dir=$(find_backup_by_version "$TARGET_VERSION")
    else
        backup_dir=$(get_latest_backup)
    fi
    
    if [ -z "$backup_dir" ]; then
        log_error "No backup directory specified or found"
        exit 1
    fi
    
    # Validate backup
    if ! validate_backup "$backup_dir"; then
        log_error "Backup validation failed"
        exit 1
    fi
    
    # Confirm rollback
    confirm_rollback "$backup_dir"
    
    # Perform rollback
    perform_rollback "$backup_dir"
    
    log_success "Rollback procedure completed!"
    log_info "Application available at: http://localhost:8000"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi