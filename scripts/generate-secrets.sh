#!/bin/bash

# ============================================================================
# Secrets Generation Script for Autonomous Agent System
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SECRETS_DIR="./secrets"
BACKUP_DIR="./secrets/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$SECRETS_DIR"
mkdir -p "$BACKUP_DIR"

# Function to generate a random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Function to generate a JWT secret
generate_jwt_secret() {
    openssl rand -hex 64
}

# Function to backup existing secrets
backup_secrets() {
    if [ -d "$SECRETS_DIR" ] && [ "$(ls -A $SECRETS_DIR)" ]; then
        echo -e "${YELLOW}Backing up existing secrets...${NC}"
        tar -czf "$BACKUP_DIR/secrets_backup_$TIMESTAMP.tar.gz" -C "$SECRETS_DIR" . --exclude="backups"
        echo -e "${GREEN}Secrets backed up to $BACKUP_DIR/secrets_backup_$TIMESTAMP.tar.gz${NC}"
    fi
}

# Function to generate all secrets
generate_all_secrets() {
    echo -e "${BLUE}Generating secrets for Autonomous Agent System...${NC}"
    
    # Database passwords
    echo -e "${YELLOW}Generating database passwords...${NC}"
    generate_password 32 > "$SECRETS_DIR/postgres_password.txt"
    generate_password 32 > "$SECRETS_DIR/redis_password.txt"
    
    # Application secrets
    echo -e "${YELLOW}Generating application secrets...${NC}"
    generate_jwt_secret > "$SECRETS_DIR/jwt_secret.txt"
    
    # Monitoring passwords
    echo -e "${YELLOW}Generating monitoring passwords...${NC}"
    generate_password 24 > "$SECRETS_DIR/grafana_password.txt"
    generate_password 24 > "$SECRETS_DIR/elasticsearch_password.txt"
    
    # API tokens (placeholders - to be filled manually)
    echo -e "${YELLOW}Creating API token placeholders...${NC}"
    echo "YOUR_GITHUB_TOKEN_HERE" > "$SECRETS_DIR/github_token.txt"
    
    # Gmail credentials (placeholder - to be filled manually)
    cat > "$SECRETS_DIR/gmail_credentials.json" <<EOF
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com"
}
EOF
    
    # Set proper permissions
    chmod 600 "$SECRETS_DIR"/*.txt
    chmod 600 "$SECRETS_DIR"/*.json
    
    echo -e "${GREEN}All secrets generated successfully!${NC}"
}

# Function to validate secrets
validate_secrets() {
    echo -e "${BLUE}Validating secrets...${NC}"
    
    local required_files=(
        "postgres_password.txt"
        "redis_password.txt"
        "jwt_secret.txt"
        "grafana_password.txt"
        "elasticsearch_password.txt"
        "github_token.txt"
        "gmail_credentials.json"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$SECRETS_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        echo -e "${GREEN}All required secrets are present.${NC}"
    else
        echo -e "${RED}Missing secrets:${NC}"
        for file in "${missing_files[@]}"; do
            echo -e "  ${RED}- $file${NC}"
        done
        exit 1
    fi
}

# Function to show secrets summary
show_summary() {
    echo -e "\n${BLUE}Secrets Summary:${NC}"
    echo -e "=================="
    
    echo -e "${GREEN}Database Passwords:${NC}"
    echo -e "  - PostgreSQL: $(wc -c < "$SECRETS_DIR/postgres_password.txt" | tr -d ' ') characters"
    echo -e "  - Redis: $(wc -c < "$SECRETS_DIR/redis_password.txt" | tr -d ' ') characters"
    
    echo -e "\n${GREEN}Application Secrets:${NC}"
    echo -e "  - JWT Secret: $(wc -c < "$SECRETS_DIR/jwt_secret.txt" | tr -d ' ') characters"
    
    echo -e "\n${GREEN}Monitoring Passwords:${NC}"
    echo -e "  - Grafana: $(wc -c < "$SECRETS_DIR/grafana_password.txt" | tr -d ' ') characters"
    echo -e "  - Elasticsearch: $(wc -c < "$SECRETS_DIR/elasticsearch_password.txt" | tr -d ' ') characters"
    
    echo -e "\n${YELLOW}Manual Configuration Required:${NC}"
    echo -e "  - GitHub Token: Update $SECRETS_DIR/github_token.txt"
    echo -e "  - Gmail Credentials: Update $SECRETS_DIR/gmail_credentials.json"
    
    echo -e "\n${BLUE}Security Notes:${NC}"
    echo -e "  - All secrets have 600 permissions (owner read/write only)"
    echo -e "  - Secrets are excluded from Git via .gitignore"
    echo -e "  - Backup created at: $BACKUP_DIR/secrets_backup_$TIMESTAMP.tar.gz"
}

# Main execution
main() {
    echo -e "${BLUE}Autonomous Agent System - Secrets Generator${NC}"
    echo -e "============================================="
    
    # Check if OpenSSL is available
    if ! command -v openssl &> /dev/null; then
        echo -e "${RED}Error: OpenSSL is required but not installed.${NC}"
        exit 1
    fi
    
    # Backup existing secrets if they exist
    backup_secrets
    
    # Generate all secrets
    generate_all_secrets
    
    # Validate secrets
    validate_secrets
    
    # Show summary
    show_summary
    
    echo -e "\n${GREEN}Secrets generation completed successfully!${NC}"
    echo -e "${YELLOW}Don't forget to update the manual configuration files.${NC}"
}

# Handle command line arguments
case "${1:-}" in
    "generate")
        main
        ;;
    "validate")
        validate_secrets
        ;;
    "backup")
        backup_secrets
        ;;
    *)
        echo "Usage: $0 {generate|validate|backup}"
        echo "  generate  - Generate all secrets"
        echo "  validate  - Validate existing secrets"
        echo "  backup    - Backup existing secrets"
        exit 1
        ;;
esac