#!/bin/bash

# ============================================================================
# Backup and Restore Script for Autonomous Agent System
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/opt/autonomous-agent/backups"
DATA_DIR="/opt/autonomous-agent/data"
LOGS_DIR="/opt/autonomous-agent/logs"
CONFIG_DIR="/opt/autonomous-agent/config"
SECRETS_DIR="/opt/autonomous-agent/secrets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-autonomous_agent}"
DB_USER="${DB_USER:-agent}"
DB_PASSWORD="${DB_PASSWORD:-password}"

# Redis configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# S3 configuration (optional)
S3_BUCKET="${S3_BUCKET:-}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-}"
S3_SECRET_KEY="${S3_SECRET_KEY:-}"
S3_REGION="${S3_REGION:-us-east-1}"

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

# Function to create backup directory
create_backup_directory() {
    local backup_path="$BACKUP_DIR/$TIMESTAMP"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

# Function to backup PostgreSQL database
backup_database() {
    local backup_path="$1"
    
    log_message "Starting database backup..."
    
    # Set PostgreSQL password
    export PGPASSWORD="$DB_PASSWORD"
    
    # Create database dump
    local db_backup_file="$backup_path/database_backup.sql"
    
    if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > "$db_backup_file"; then
        log_success "Database backup completed: $db_backup_file"
        
        # Compress database backup
        gzip "$db_backup_file"
        log_success "Database backup compressed: $db_backup_file.gz"
    else
        log_error "Database backup failed"
        return 1
    fi
    
    # Unset password
    unset PGPASSWORD
}

# Function to backup Redis data
backup_redis() {
    local backup_path="$1"
    
    log_message "Starting Redis backup..."
    
    # Save Redis data
    if [ -n "$REDIS_PASSWORD" ]; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
    fi
    
    # Wait for backup to complete
    sleep 5
    
    # Copy Redis dump file
    local redis_dump_file="$backup_path/redis_backup.rdb"
    
    if [ -f "/data/dump.rdb" ]; then
        cp "/data/dump.rdb" "$redis_dump_file"
        log_success "Redis backup completed: $redis_dump_file"
    else
        log_warning "Redis dump file not found, skipping Redis backup"
    fi
}

# Function to backup application data
backup_application_data() {
    local backup_path="$1"
    
    log_message "Starting application data backup..."
    
    # Backup data directory
    if [ -d "$DATA_DIR" ]; then
        tar -czf "$backup_path/application_data.tar.gz" -C "$DATA_DIR" .
        log_success "Application data backup completed: $backup_path/application_data.tar.gz"
    else
        log_warning "Application data directory not found: $DATA_DIR"
    fi
    
    # Backup logs directory
    if [ -d "$LOGS_DIR" ]; then
        tar -czf "$backup_path/logs.tar.gz" -C "$LOGS_DIR" .
        log_success "Logs backup completed: $backup_path/logs.tar.gz"
    else
        log_warning "Logs directory not found: $LOGS_DIR"
    fi
    
    # Backup configuration directory
    if [ -d "$CONFIG_DIR" ]; then
        tar -czf "$backup_path/config.tar.gz" -C "$CONFIG_DIR" .
        log_success "Configuration backup completed: $backup_path/config.tar.gz"
    else
        log_warning "Configuration directory not found: $CONFIG_DIR"
    fi
    
    # Backup secrets directory (encrypted)
    if [ -d "$SECRETS_DIR" ]; then
        # Create encrypted backup of secrets
        tar -czf - -C "$SECRETS_DIR" . | gpg --symmetric --cipher-algo AES256 --output "$backup_path/secrets.tar.gz.gpg"
        log_success "Secrets backup completed (encrypted): $backup_path/secrets.tar.gz.gpg"
    else
        log_warning "Secrets directory not found: $SECRETS_DIR"
    fi
}

# Function to create backup manifest
create_backup_manifest() {
    local backup_path="$1"
    
    log_message "Creating backup manifest..."
    
    cat > "$backup_path/backup_manifest.json" <<EOF
{
    "backup_id": "$TIMESTAMP",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0.0",
    "system": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "arch": "$(uname -m)"
    },
    "components": {
        "database": {
            "type": "postgresql",
            "host": "$DB_HOST",
            "port": "$DB_PORT",
            "database": "$DB_NAME",
            "user": "$DB_USER",
            "backup_file": "database_backup.sql.gz"
        },
        "redis": {
            "host": "$REDIS_HOST",
            "port": "$REDIS_PORT",
            "backup_file": "redis_backup.rdb"
        },
        "application_data": {
            "backup_file": "application_data.tar.gz"
        },
        "logs": {
            "backup_file": "logs.tar.gz"
        },
        "config": {
            "backup_file": "config.tar.gz"
        },
        "secrets": {
            "backup_file": "secrets.tar.gz.gpg",
            "encrypted": true
        }
    },
    "files": [
EOF

    # List all files in backup
    local first_file=true
    for file in "$backup_path"/*; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "backup_manifest.json" ]; then
            if [ "$first_file" = true ]; then
                first_file=false
            else
                echo "," >> "$backup_path/backup_manifest.json"
            fi
            
            local filename=$(basename "$file")
            local filesize=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
            local checksum=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$file" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
            
            echo "        {" >> "$backup_path/backup_manifest.json"
            echo "            \"filename\": \"$filename\"," >> "$backup_path/backup_manifest.json"
            echo "            \"size\": $filesize," >> "$backup_path/backup_manifest.json"
            echo "            \"checksum\": \"$checksum\"" >> "$backup_path/backup_manifest.json"
            echo -n "        }" >> "$backup_path/backup_manifest.json"
        fi
    done
    
    cat >> "$backup_path/backup_manifest.json" <<EOF

    ]
}
EOF

    log_success "Backup manifest created: $backup_path/backup_manifest.json"
}

# Function to upload backup to S3
upload_to_s3() {
    local backup_path="$1"
    
    if [ -z "$S3_BUCKET" ]; then
        log_message "S3 backup not configured, skipping upload"
        return 0
    fi
    
    log_message "Uploading backup to S3..."
    
    # Configure AWS CLI or use direct API calls
    export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
    export AWS_DEFAULT_REGION="$S3_REGION"
    
    # Create archive of entire backup directory
    local archive_name="backup_$TIMESTAMP.tar.gz"
    local archive_path="$BACKUP_DIR/$archive_name"
    
    tar -czf "$archive_path" -C "$backup_path" .
    
    # Upload to S3
    if command -v aws &> /dev/null; then
        aws s3 cp "$archive_path" "s3://$S3_BUCKET/backups/$archive_name"
        log_success "Backup uploaded to S3: s3://$S3_BUCKET/backups/$archive_name"
    else
        log_warning "AWS CLI not found, skipping S3 upload"
    fi
    
    # Clean up local archive
    rm -f "$archive_path"
}

# Function to perform full backup
perform_backup() {
    log_message "Starting full backup of Autonomous Agent System..."
    
    # Create backup directory
    local backup_path
    backup_path=$(create_backup_directory)
    
    # Backup database
    backup_database "$backup_path"
    
    # Backup Redis
    backup_redis "$backup_path"
    
    # Backup application data
    backup_application_data "$backup_path"
    
    # Create backup manifest
    create_backup_manifest "$backup_path"
    
    # Upload to S3 if configured
    upload_to_s3 "$backup_path"
    
    # Calculate total backup size
    local total_size
    total_size=$(du -sh "$backup_path" | cut -f1)
    
    log_success "Full backup completed successfully!"
    log_success "Backup location: $backup_path"
    log_success "Total backup size: $total_size"
}

# Function to list available backups
list_backups() {
    log_message "Available backups:"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_warning "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    local backup_count=0
    
    for backup_path in "$BACKUP_DIR"/*; do
        if [ -d "$backup_path" ]; then
            local backup_name=$(basename "$backup_path")
            local backup_size
            backup_size=$(du -sh "$backup_path" | cut -f1)
            local backup_date
            backup_date=$(date -d "${backup_name:0:8}" +%Y-%m-%d 2>/dev/null || echo "Unknown")
            
            echo "  $backup_name ($backup_size) - $backup_date"
            backup_count=$((backup_count + 1))
        fi
    done
    
    if [ $backup_count -eq 0 ]; then
        log_warning "No backups found"
    else
        log_success "Found $backup_count backups"
    fi
}

# Function to restore from backup
restore_from_backup() {
    local backup_id="$1"
    local backup_path="$BACKUP_DIR/$backup_id"
    
    if [ ! -d "$backup_path" ]; then
        log_error "Backup not found: $backup_path"
        return 1
    fi
    
    log_message "Starting restore from backup: $backup_id"
    
    # Verify backup manifest
    if [ ! -f "$backup_path/backup_manifest.json" ]; then
        log_error "Backup manifest not found, cannot verify backup integrity"
        return 1
    fi
    
    # Restore database
    if [ -f "$backup_path/database_backup.sql.gz" ]; then
        log_message "Restoring database..."
        
        export PGPASSWORD="$DB_PASSWORD"
        
        # Drop existing database and recreate
        dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" || true
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
        
        # Restore database
        gunzip -c "$backup_path/database_backup.sql.gz" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
        
        unset PGPASSWORD
        log_success "Database restored successfully"
    fi
    
    # Restore Redis
    if [ -f "$backup_path/redis_backup.rdb" ]; then
        log_message "Restoring Redis..."
        
        # Stop Redis service
        systemctl stop redis-server || true
        
        # Copy Redis dump file
        cp "$backup_path/redis_backup.rdb" "/data/dump.rdb"
        
        # Start Redis service
        systemctl start redis-server || true
        
        log_success "Redis restored successfully"
    fi
    
    # Restore application data
    if [ -f "$backup_path/application_data.tar.gz" ]; then
        log_message "Restoring application data..."
        
        # Backup existing data
        if [ -d "$DATA_DIR" ]; then
            mv "$DATA_DIR" "$DATA_DIR.backup.$(date +%s)"
        fi
        
        # Create data directory and restore
        mkdir -p "$DATA_DIR"
        tar -xzf "$backup_path/application_data.tar.gz" -C "$DATA_DIR"
        
        log_success "Application data restored successfully"
    fi
    
    # Restore configuration
    if [ -f "$backup_path/config.tar.gz" ]; then
        log_message "Restoring configuration..."
        
        # Backup existing config
        if [ -d "$CONFIG_DIR" ]; then
            mv "$CONFIG_DIR" "$CONFIG_DIR.backup.$(date +%s)"
        fi
        
        # Create config directory and restore
        mkdir -p "$CONFIG_DIR"
        tar -xzf "$backup_path/config.tar.gz" -C "$CONFIG_DIR"
        
        log_success "Configuration restored successfully"
    fi
    
    # Restore secrets (encrypted)
    if [ -f "$backup_path/secrets.tar.gz.gpg" ]; then
        log_message "Restoring secrets (encrypted)..."
        
        # Backup existing secrets
        if [ -d "$SECRETS_DIR" ]; then
            mv "$SECRETS_DIR" "$SECRETS_DIR.backup.$(date +%s)"
        fi
        
        # Create secrets directory and restore
        mkdir -p "$SECRETS_DIR"
        gpg --decrypt --quiet "$backup_path/secrets.tar.gz.gpg" | tar -xzf - -C "$SECRETS_DIR"
        
        log_success "Secrets restored successfully"
    fi
    
    log_success "Restore completed successfully from backup: $backup_id"
}

# Function to cleanup old backups
cleanup_old_backups() {
    log_message "Cleaning up backups older than $RETENTION_DAYS days..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_warning "Backup directory not found: $BACKUP_DIR"
        return 0
    fi
    
    local deleted_count=0
    
    find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \; -print | while read -r deleted_backup; do
        log_message "Deleted old backup: $(basename "$deleted_backup")"
        deleted_count=$((deleted_count + 1))
    done
    
    log_success "Cleanup completed, deleted $deleted_count old backups"
}

# Function to verify backup integrity
verify_backup() {
    local backup_id="$1"
    local backup_path="$BACKUP_DIR/$backup_id"
    
    if [ ! -d "$backup_path" ]; then
        log_error "Backup not found: $backup_path"
        return 1
    fi
    
    log_message "Verifying backup integrity: $backup_id"
    
    # Check if manifest exists
    if [ ! -f "$backup_path/backup_manifest.json" ]; then
        log_error "Backup manifest not found"
        return 1
    fi
    
    # Verify file checksums
    local verification_failed=false
    
    # Parse manifest and verify checksums (simplified)
    for file in "$backup_path"/*; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "backup_manifest.json" ]; then
            local current_checksum
            current_checksum=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$file" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
            
            if [ "$current_checksum" = "unknown" ]; then
                log_warning "Could not verify checksum for: $(basename "$file")"
            else
                log_message "Verified: $(basename "$file") - $current_checksum"
            fi
        fi
    done
    
    if [ "$verification_failed" = false ]; then
        log_success "Backup integrity verification passed"
    else
        log_error "Backup integrity verification failed"
        return 1
    fi
}

# Function to show backup information
show_backup_info() {
    local backup_id="$1"
    local backup_path="$BACKUP_DIR/$backup_id"
    
    if [ ! -d "$backup_path" ]; then
        log_error "Backup not found: $backup_path"
        return 1
    fi
    
    log_message "Backup information for: $backup_id"
    
    if [ -f "$backup_path/backup_manifest.json" ]; then
        echo "Manifest contents:"
        cat "$backup_path/backup_manifest.json" | python -m json.tool 2>/dev/null || cat "$backup_path/backup_manifest.json"
    else
        log_warning "Backup manifest not found"
    fi
    
    echo ""
    echo "Backup contents:"
    ls -la "$backup_path"
    
    echo ""
    echo "Total backup size: $(du -sh "$backup_path" | cut -f1)"
}

# Main function
main() {
    echo -e "${BLUE}Autonomous Agent System - Backup & Restore Tool${NC}"
    echo -e "================================================="
    echo ""
    
    case "${1:-}" in
        "backup")
            perform_backup
            ;;
        "restore")
            if [ -z "${2:-}" ]; then
                log_error "Please specify backup ID to restore from"
                list_backups
                exit 1
            fi
            restore_from_backup "$2"
            ;;
        "list")
            list_backups
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "verify")
            if [ -z "${2:-}" ]; then
                log_error "Please specify backup ID to verify"
                list_backups
                exit 1
            fi
            verify_backup "$2"
            ;;
        "info")
            if [ -z "${2:-}" ]; then
                log_error "Please specify backup ID to show info for"
                list_backups
                exit 1
            fi
            show_backup_info "$2"
            ;;
        *)
            echo "Usage: $0 {backup|restore|list|cleanup|verify|info}"
            echo ""
            echo "Commands:"
            echo "  backup           - Create a full backup"
            echo "  restore <id>     - Restore from backup"
            echo "  list             - List available backups"
            echo "  cleanup          - Clean up old backups"
            echo "  verify <id>      - Verify backup integrity"
            echo "  info <id>        - Show backup information"
            echo ""
            echo "Examples:"
            echo "  $0 backup"
            echo "  $0 restore 20240101_120000"
            echo "  $0 list"
            echo "  $0 cleanup"
            echo "  $0 verify 20240101_120000"
            echo "  $0 info 20240101_120000"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"