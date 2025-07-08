#!/bin/bash

# ============================================================================
# Security Compliance Validation Script
# ============================================================================
# This script performs comprehensive security compliance checks for the
# autonomous agent system according to industry standards and best practices.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPLIANCE_REPORT_DIR="${PROJECT_ROOT}/compliance-reports"
COMPLIANCE_STANDARDS="${COMPLIANCE_STANDARDS:-SOC2,ISO27001,NIST,OWASP}"
SEVERITY_THRESHOLD="${SEVERITY_THRESHOLD:-medium}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compliance results
COMPLIANCE_PASSED=0
COMPLIANCE_FAILED=0
COMPLIANCE_WARNINGS=0

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
    -s, --standards LIST     Compliance standards to check (SOC2,ISO27001,NIST,OWASP)
    -t, --threshold LEVEL    Severity threshold (low, medium, high, critical)
    -r, --report-dir DIR     Directory for compliance reports
    -f, --format FORMAT      Report format (text, json, html, xml)
    -c, --category CATEGORY  Check specific category only
    -v, --verbose            Verbose output
    -h, --help               Show this help message

Standards:
    SOC2        - SOC 2 Type II compliance
    ISO27001    - ISO 27001 Information Security Management
    NIST        - NIST Cybersecurity Framework
    OWASP       - OWASP Top 10 Security Risks
    PCI         - PCI DSS (if applicable)
    HIPAA       - HIPAA compliance (if applicable)

Categories:
    access-control       - Access control and authentication
    data-protection      - Data encryption and protection
    network-security     - Network security configuration
    logging-monitoring   - Logging and monitoring
    incident-response    - Incident response procedures
    backup-recovery      - Backup and recovery processes
    configuration        - Security configuration management

Examples:
    $0 --standards SOC2,NIST
    $0 --category access-control --threshold high
    $0 --format html --report-dir /tmp/compliance
EOF
}

# Parse command line arguments
REPORT_FORMAT="text"
CATEGORY=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--standards)
            COMPLIANCE_STANDARDS="$2"
            shift 2
            ;;
        -t|--threshold)
            SEVERITY_THRESHOLD="$2"
            shift 2
            ;;
        -r|--report-dir)
            COMPLIANCE_REPORT_DIR="$2"
            shift 2
            ;;
        -f|--format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
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

# Initialize compliance environment
initialize_compliance() {
    log_info "Initializing compliance check environment..."
    
    # Create report directory
    mkdir -p "$COMPLIANCE_REPORT_DIR"
    
    # Create compliance report header
    cat > "$COMPLIANCE_REPORT_DIR/compliance-report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Compliance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .passed { color: green; }
        .failed { color: red; }
        .warning { color: orange; }
        .critical { color: darkred; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .standard-section { margin: 20px 0; }
        .check-result { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .check-result.passed { border-left-color: green; }
        .check-result.failed { border-left-color: red; }
        .check-result.warning { border-left-color: orange; }
    </style>
</head>
<body>
    <h1>Security Compliance Report</h1>
    <p>Generated: $(date)</p>
    <p>Standards: $COMPLIANCE_STANDARDS</p>
    <p>Severity Threshold: $SEVERITY_THRESHOLD</p>
    <div id="results">
EOF
    
    log_success "Compliance environment initialized"
}

# Record compliance result
record_compliance_result() {
    local check_name="$1"
    local status="$2"
    local severity="$3"
    local description="$4"
    local remediation="$5"
    local standard="$6"
    
    case "$status" in
        "PASSED")
            COMPLIANCE_PASSED=$((COMPLIANCE_PASSED + 1))
            if [ "$VERBOSE" = true ]; then
                log_success "$check_name: $description"
            fi
            ;;
        "FAILED")
            COMPLIANCE_FAILED=$((COMPLIANCE_FAILED + 1))
            log_error "$check_name: $description"
            ;;
        "WARNING")
            COMPLIANCE_WARNINGS=$((COMPLIANCE_WARNINGS + 1))
            log_warning "$check_name: $description"
            ;;
    esac
    
    # Add to HTML report
    cat >> "$COMPLIANCE_REPORT_DIR/compliance-report.html" << EOF
    <div class="check-result $status">
        <h3>$check_name</h3>
        <p><strong>Status:</strong> <span class="$(echo $status | tr '[:upper:]' '[:lower:]')">$status</span></p>
        <p><strong>Severity:</strong> $severity</p>
        <p><strong>Standard:</strong> $standard</p>
        <p><strong>Description:</strong> $description</p>
        <p><strong>Remediation:</strong> $remediation</p>
    </div>
EOF
}

# Access Control and Authentication Checks
check_access_control() {
    log_info "Checking access control and authentication..."
    
    # Check JWT configuration
    if grep -q "JWT_SECRET" "$PROJECT_ROOT/.env.production" 2>/dev/null; then
        local jwt_secret=$(grep "JWT_SECRET" "$PROJECT_ROOT/.env.production" | cut -d'=' -f2)
        if [ ${#jwt_secret} -ge 32 ]; then
            record_compliance_result "JWT Secret Strength" "PASSED" "high" "JWT secret is sufficiently strong" "None required" "SOC2,ISO27001"
        else
            record_compliance_result "JWT Secret Strength" "FAILED" "high" "JWT secret is too weak" "Use a strong, randomly generated secret of at least 32 characters" "SOC2,ISO27001"
        fi
    else
        record_compliance_result "JWT Configuration" "FAILED" "critical" "JWT secret not configured" "Configure JWT_SECRET in environment variables" "SOC2,ISO27001"
    fi
    
    # Check password policies
    if [ -f "$PROJECT_ROOT/config/security.yml" ]; then
        if grep -q "password_policy" "$PROJECT_ROOT/config/security.yml"; then
            record_compliance_result "Password Policy" "PASSED" "medium" "Password policy is configured" "None required" "SOC2,ISO27001,NIST"
        else
            record_compliance_result "Password Policy" "WARNING" "medium" "Password policy not explicitly configured" "Define password complexity requirements" "SOC2,ISO27001,NIST"
        fi
    else
        record_compliance_result "Security Configuration" "FAILED" "high" "Security configuration file not found" "Create security configuration file" "SOC2,ISO27001,NIST"
    fi
    
    # Check multi-factor authentication
    if grep -q "mfa_enabled" "$PROJECT_ROOT/config/security.yml" 2>/dev/null; then
        record_compliance_result "Multi-Factor Authentication" "PASSED" "high" "MFA is configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Multi-Factor Authentication" "WARNING" "high" "MFA configuration not found" "Implement multi-factor authentication" "SOC2,ISO27001,NIST"
    fi
    
    # Check session management
    if grep -q "session_timeout" "$PROJECT_ROOT/config/security.yml" 2>/dev/null; then
        record_compliance_result "Session Management" "PASSED" "medium" "Session timeout is configured" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Session Management" "WARNING" "medium" "Session timeout not configured" "Configure session timeout and management" "SOC2,ISO27001"
    fi
    
    # Check role-based access control
    if [ -f "$PROJECT_ROOT/src/auth/rbac.py" ] || [ -f "$PROJECT_ROOT/src/auth/roles.py" ]; then
        record_compliance_result "Role-Based Access Control" "PASSED" "high" "RBAC implementation found" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Role-Based Access Control" "WARNING" "high" "RBAC implementation not found" "Implement role-based access control" "SOC2,ISO27001"
    fi
}

# Data Protection Checks
check_data_protection() {
    log_info "Checking data protection and encryption..."
    
    # Check encryption configuration
    if grep -q "ENCRYPTION_KEY" "$PROJECT_ROOT/.env.production" 2>/dev/null; then
        record_compliance_result "Data Encryption" "PASSED" "high" "Encryption key is configured" "None required" "SOC2,ISO27001,PCI"
    else
        record_compliance_result "Data Encryption" "FAILED" "high" "Encryption key not configured" "Configure ENCRYPTION_KEY for data encryption" "SOC2,ISO27001,PCI"
    fi
    
    # Check database encryption
    if grep -q "ssl_mode" "$PROJECT_ROOT/docker/postgres/postgresql.conf" 2>/dev/null; then
        record_compliance_result "Database Encryption" "PASSED" "high" "Database SSL is configured" "None required" "SOC2,ISO27001,PCI"
    else
        record_compliance_result "Database Encryption" "WARNING" "high" "Database SSL not configured" "Enable SSL for database connections" "SOC2,ISO27001,PCI"
    fi
    
    # Check data at rest encryption
    if grep -q "encryption" "$PROJECT_ROOT/docker-compose.prod.yml" 2>/dev/null; then
        record_compliance_result "Data at Rest Encryption" "PASSED" "high" "Data at rest encryption configured" "None required" "SOC2,ISO27001,PCI"
    else
        record_compliance_result "Data at Rest Encryption" "WARNING" "high" "Data at rest encryption not configured" "Enable encryption for data at rest" "SOC2,ISO27001,PCI"
    fi
    
    # Check secrets management
    if [ -d "$PROJECT_ROOT/secrets" ] && [ "$(stat -c %a "$PROJECT_ROOT/secrets")" = "700" ]; then
        record_compliance_result "Secrets Management" "PASSED" "high" "Secrets directory has correct permissions" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Secrets Management" "FAILED" "high" "Secrets directory permissions incorrect" "Set secrets directory permissions to 700" "SOC2,ISO27001"
    fi
    
    # Check data retention policies
    if [ -f "$PROJECT_ROOT/config/data-retention.yml" ]; then
        record_compliance_result "Data Retention Policy" "PASSED" "medium" "Data retention policy is configured" "None required" "SOC2,ISO27001,HIPAA"
    else
        record_compliance_result "Data Retention Policy" "WARNING" "medium" "Data retention policy not configured" "Define data retention policies" "SOC2,ISO27001,HIPAA"
    fi
}

# Network Security Checks
check_network_security() {
    log_info "Checking network security configuration..."
    
    # Check HTTPS configuration
    if grep -q "ssl_certificate" "$PROJECT_ROOT/config/templates/nginx.conf" 2>/dev/null; then
        record_compliance_result "HTTPS Configuration" "PASSED" "high" "HTTPS is configured" "None required" "SOC2,ISO27001,OWASP"
    else
        record_compliance_result "HTTPS Configuration" "FAILED" "high" "HTTPS not configured" "Configure SSL/TLS certificates" "SOC2,ISO27001,OWASP"
    fi
    
    # Check security headers
    if grep -q "X-Frame-Options" "$PROJECT_ROOT/config/templates/nginx.conf" 2>/dev/null; then
        record_compliance_result "Security Headers" "PASSED" "medium" "Security headers are configured" "None required" "OWASP"
    else
        record_compliance_result "Security Headers" "WARNING" "medium" "Security headers not configured" "Configure security headers in web server" "OWASP"
    fi
    
    # Check network segmentation
    if grep -q "internal: true" "$PROJECT_ROOT/docker-compose.prod.yml" 2>/dev/null; then
        record_compliance_result "Network Segmentation" "PASSED" "high" "Network segmentation is configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Network Segmentation" "WARNING" "high" "Network segmentation not configured" "Implement network segmentation" "SOC2,ISO27001,NIST"
    fi
    
    # Check firewall rules
    if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
        record_compliance_result "Firewall Configuration" "PASSED" "high" "Firewall is active" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Firewall Configuration" "WARNING" "high" "Firewall not active or not configured" "Configure and enable firewall" "SOC2,ISO27001,NIST"
    fi
    
    # Check rate limiting
    if grep -q "limit_req" "$PROJECT_ROOT/config/templates/nginx.conf" 2>/dev/null; then
        record_compliance_result "Rate Limiting" "PASSED" "medium" "Rate limiting is configured" "None required" "OWASP"
    else
        record_compliance_result "Rate Limiting" "WARNING" "medium" "Rate limiting not configured" "Implement rate limiting" "OWASP"
    fi
}

# Logging and Monitoring Checks
check_logging_monitoring() {
    log_info "Checking logging and monitoring configuration..."
    
    # Check audit logging
    if grep -q "audit_log" "$PROJECT_ROOT/config/production/app.yml" 2>/dev/null; then
        record_compliance_result "Audit Logging" "PASSED" "high" "Audit logging is configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Audit Logging" "WARNING" "high" "Audit logging not configured" "Enable audit logging" "SOC2,ISO27001,NIST"
    fi
    
    # Check log retention
    if grep -q "log_retention" "$PROJECT_ROOT/config/production/app.yml" 2>/dev/null; then
        record_compliance_result "Log Retention" "PASSED" "medium" "Log retention is configured" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Log Retention" "WARNING" "medium" "Log retention not configured" "Configure log retention policies" "SOC2,ISO27001"
    fi
    
    # Check monitoring alerts
    if [ -f "$PROJECT_ROOT/config/monitoring/alerts.yml" ]; then
        record_compliance_result "Security Monitoring" "PASSED" "high" "Security monitoring alerts configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Security Monitoring" "WARNING" "high" "Security monitoring not configured" "Configure security monitoring and alerts" "SOC2,ISO27001,NIST"
    fi
    
    # Check log protection
    if grep -q "syslog" "$PROJECT_ROOT/config/production/app.yml" 2>/dev/null; then
        record_compliance_result "Log Protection" "PASSED" "medium" "Centralized logging is configured" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Log Protection" "WARNING" "medium" "Centralized logging not configured" "Configure centralized logging" "SOC2,ISO27001"
    fi
}

# Incident Response Checks
check_incident_response() {
    log_info "Checking incident response procedures..."
    
    # Check incident response plan
    if [ -f "$PROJECT_ROOT/docs/incident-response.md" ]; then
        record_compliance_result "Incident Response Plan" "PASSED" "high" "Incident response plan exists" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Incident Response Plan" "WARNING" "high" "Incident response plan not found" "Create incident response plan" "SOC2,ISO27001,NIST"
    fi
    
    # Check emergency contacts
    if [ -f "$PROJECT_ROOT/docs/emergency-contacts.md" ]; then
        record_compliance_result "Emergency Contacts" "PASSED" "medium" "Emergency contacts documented" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Emergency Contacts" "WARNING" "medium" "Emergency contacts not documented" "Document emergency contacts" "SOC2,ISO27001"
    fi
    
    # Check incident logging
    if grep -q "incident_log" "$PROJECT_ROOT/config/production/app.yml" 2>/dev/null; then
        record_compliance_result "Incident Logging" "PASSED" "high" "Incident logging is configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Incident Logging" "WARNING" "high" "Incident logging not configured" "Configure incident logging" "SOC2,ISO27001,NIST"
    fi
}

# Backup and Recovery Checks
check_backup_recovery() {
    log_info "Checking backup and recovery procedures..."
    
    # Check backup configuration
    if [ -f "$PROJECT_ROOT/scripts/backup-restore.sh" ]; then
        record_compliance_result "Backup Procedures" "PASSED" "high" "Backup procedures are documented" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Backup Procedures" "WARNING" "high" "Backup procedures not documented" "Document backup procedures" "SOC2,ISO27001,NIST"
    fi
    
    # Check backup encryption
    if grep -q "encryption" "$PROJECT_ROOT/scripts/backup-restore.sh" 2>/dev/null; then
        record_compliance_result "Backup Encryption" "PASSED" "high" "Backup encryption is configured" "None required" "SOC2,ISO27001,PCI"
    else
        record_compliance_result "Backup Encryption" "WARNING" "high" "Backup encryption not configured" "Enable backup encryption" "SOC2,ISO27001,PCI"
    fi
    
    # Check disaster recovery plan
    if [ -f "$PROJECT_ROOT/scripts/production/disaster-recovery.sh" ]; then
        record_compliance_result "Disaster Recovery Plan" "PASSED" "high" "Disaster recovery plan exists" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Disaster Recovery Plan" "WARNING" "high" "Disaster recovery plan not found" "Create disaster recovery plan" "SOC2,ISO27001,NIST"
    fi
    
    # Check backup testing
    if [ -f "$PROJECT_ROOT/scripts/test-backup.sh" ]; then
        record_compliance_result "Backup Testing" "PASSED" "medium" "Backup testing procedures exist" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Backup Testing" "WARNING" "medium" "Backup testing not configured" "Create backup testing procedures" "SOC2,ISO27001"
    fi
}

# Configuration Management Checks
check_configuration_management() {
    log_info "Checking configuration management..."
    
    # Check configuration version control
    if [ -d "$PROJECT_ROOT/.git" ]; then
        record_compliance_result "Configuration Version Control" "PASSED" "medium" "Configuration is version controlled" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Configuration Version Control" "WARNING" "medium" "Configuration not version controlled" "Use version control for configuration" "SOC2,ISO27001,NIST"
    fi
    
    # Check configuration validation
    if [ -f "$PROJECT_ROOT/scripts/validate-config.sh" ]; then
        record_compliance_result "Configuration Validation" "PASSED" "medium" "Configuration validation exists" "None required" "SOC2,ISO27001"
    else
        record_compliance_result "Configuration Validation" "WARNING" "medium" "Configuration validation not found" "Create configuration validation scripts" "SOC2,ISO27001"
    fi
    
    # Check security hardening
    if grep -q "security_hardening" "$PROJECT_ROOT/config/production/app.yml" 2>/dev/null; then
        record_compliance_result "Security Hardening" "PASSED" "high" "Security hardening is configured" "None required" "SOC2,ISO27001,NIST"
    else
        record_compliance_result "Security Hardening" "WARNING" "high" "Security hardening not configured" "Implement security hardening" "SOC2,ISO27001,NIST"
    fi
}

# OWASP Top 10 Checks
check_owasp_top10() {
    log_info "Checking OWASP Top 10 security risks..."
    
    # A01: Broken Access Control
    if grep -q "authorization" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A01 - Access Control" "PASSED" "high" "Access control implementation found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A01 - Access Control" "WARNING" "high" "Access control implementation not found" "Implement proper access control" "OWASP"
    fi
    
    # A02: Cryptographic Failures
    if grep -q "encrypt\|crypto\|hash" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A02 - Cryptographic Failures" "PASSED" "high" "Cryptographic functions found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A02 - Cryptographic Failures" "WARNING" "high" "Cryptographic implementation not found" "Implement proper cryptography" "OWASP"
    fi
    
    # A03: Injection
    if grep -q "sql.*injection\|parameterized" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A03 - Injection" "PASSED" "high" "Injection protection found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A03 - Injection" "WARNING" "high" "Injection protection not found" "Implement input validation and parameterized queries" "OWASP"
    fi
    
    # A04: Insecure Design
    if [ -f "$PROJECT_ROOT/docs/security-design.md" ]; then
        record_compliance_result "OWASP A04 - Insecure Design" "PASSED" "medium" "Security design documentation found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A04 - Insecure Design" "WARNING" "medium" "Security design documentation not found" "Document security design principles" "OWASP"
    fi
    
    # A05: Security Misconfiguration
    if [ -f "$PROJECT_ROOT/scripts/security-config-check.sh" ]; then
        record_compliance_result "OWASP A05 - Security Misconfiguration" "PASSED" "high" "Security configuration check exists" "None required" "OWASP"
    else
        record_compliance_result "OWASP A05 - Security Misconfiguration" "WARNING" "high" "Security configuration check not found" "Create security configuration checks" "OWASP"
    fi
    
    # A06: Vulnerable Components
    if [ -f "$PROJECT_ROOT/scripts/dependency-check.sh" ]; then
        record_compliance_result "OWASP A06 - Vulnerable Components" "PASSED" "high" "Dependency check exists" "None required" "OWASP"
    else
        record_compliance_result "OWASP A06 - Vulnerable Components" "WARNING" "high" "Dependency check not found" "Implement dependency vulnerability scanning" "OWASP"
    fi
    
    # A07: Identification and Authentication Failures
    if grep -q "authentication\|session" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A07 - Authentication Failures" "PASSED" "high" "Authentication implementation found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A07 - Authentication Failures" "WARNING" "high" "Authentication implementation not found" "Implement proper authentication" "OWASP"
    fi
    
    # A08: Software and Data Integrity Failures
    if [ -f "$PROJECT_ROOT/scripts/integrity-check.sh" ]; then
        record_compliance_result "OWASP A08 - Data Integrity Failures" "PASSED" "medium" "Integrity check exists" "None required" "OWASP"
    else
        record_compliance_result "OWASP A08 - Data Integrity Failures" "WARNING" "medium" "Integrity check not found" "Implement data integrity checks" "OWASP"
    fi
    
    # A09: Security Logging and Monitoring Failures
    if grep -q "security.*log\|audit" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A09 - Logging Failures" "PASSED" "high" "Security logging implementation found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A09 - Logging Failures" "WARNING" "high" "Security logging not found" "Implement security logging and monitoring" "OWASP"
    fi
    
    # A10: Server-Side Request Forgery
    if grep -q "ssrf\|request.*validation" "$PROJECT_ROOT/src/" -R 2>/dev/null; then
        record_compliance_result "OWASP A10 - SSRF" "PASSED" "medium" "SSRF protection found" "None required" "OWASP"
    else
        record_compliance_result "OWASP A10 - SSRF" "WARNING" "medium" "SSRF protection not found" "Implement SSRF protection" "OWASP"
    fi
}

# Generate compliance report
generate_compliance_report() {
    log_info "Generating compliance report..."
    
    # Close HTML report
    cat >> "$COMPLIANCE_REPORT_DIR/compliance-report.html" << EOF
    </div>
    <h2>Compliance Summary</h2>
    <table>
        <tr><th>Status</th><th>Count</th></tr>
        <tr><td class="passed">Passed</td><td>$COMPLIANCE_PASSED</td></tr>
        <tr><td class="failed">Failed</td><td>$COMPLIANCE_FAILED</td></tr>
        <tr><td class="warning">Warnings</td><td>$COMPLIANCE_WARNINGS</td></tr>
    </table>
    <p>Total Checks: $((COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS))</p>
    <p>Compliance Score: $(( COMPLIANCE_PASSED * 100 / (COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS) ))%</p>
</body>
</html>
EOF
    
    # Generate JSON report
    cat > "$COMPLIANCE_REPORT_DIR/compliance-report.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "standards": "$COMPLIANCE_STANDARDS",
    "threshold": "$SEVERITY_THRESHOLD",
    "summary": {
        "passed": $COMPLIANCE_PASSED,
        "failed": $COMPLIANCE_FAILED,
        "warnings": $COMPLIANCE_WARNINGS,
        "total": $((COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS)),
        "score": $(( COMPLIANCE_PASSED * 100 / (COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS) ))
    }
}
EOF
    
    # Generate text summary
    cat > "$COMPLIANCE_REPORT_DIR/compliance-summary.txt" << EOF
Security Compliance Report
=========================
Date: $(date)
Standards: $COMPLIANCE_STANDARDS
Threshold: $SEVERITY_THRESHOLD

Results:
- Passed: $COMPLIANCE_PASSED
- Failed: $COMPLIANCE_FAILED
- Warnings: $COMPLIANCE_WARNINGS
- Total: $((COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS))

Compliance Score: $(( COMPLIANCE_PASSED * 100 / (COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS) ))%

Recommendations:
1. Review and address all failed checks
2. Implement solutions for warning items
3. Regularly update compliance procedures
4. Conduct quarterly compliance reviews
5. Train staff on security best practices
EOF
    
    log_info "Compliance report generated: $COMPLIANCE_REPORT_DIR/"
}

# Main function
main() {
    log_info "Starting security compliance validation"
    log_info "Standards: $COMPLIANCE_STANDARDS"
    log_info "Threshold: $SEVERITY_THRESHOLD"
    
    # Initialize compliance environment
    initialize_compliance
    
    # Run compliance checks based on category or all
    if [ -n "$CATEGORY" ]; then
        case "$CATEGORY" in
            "access-control")
                check_access_control
                ;;
            "data-protection")
                check_data_protection
                ;;
            "network-security")
                check_network_security
                ;;
            "logging-monitoring")
                check_logging_monitoring
                ;;
            "incident-response")
                check_incident_response
                ;;
            "backup-recovery")
                check_backup_recovery
                ;;
            "configuration")
                check_configuration_management
                ;;
            *)
                log_error "Unknown category: $CATEGORY"
                exit 1
                ;;
        esac
    else
        # Run all checks
        check_access_control
        check_data_protection
        check_network_security
        check_logging_monitoring
        check_incident_response
        check_backup_recovery
        check_configuration_management
        
        # Run standard-specific checks
        if [[ "$COMPLIANCE_STANDARDS" == *"OWASP"* ]]; then
            check_owasp_top10
        fi
    fi
    
    # Generate compliance report
    generate_compliance_report
    
    # Print summary
    echo
    echo "=================================="
    echo "Security Compliance Summary"
    echo "=================================="
    echo "Passed:   $COMPLIANCE_PASSED"
    echo "Failed:   $COMPLIANCE_FAILED"
    echo "Warnings: $COMPLIANCE_WARNINGS"
    echo "Total:    $((COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS))"
    echo "Score:    $(( COMPLIANCE_PASSED * 100 / (COMPLIANCE_PASSED + COMPLIANCE_FAILED + COMPLIANCE_WARNINGS) ))%"
    echo "=================================="
    
    # Set exit code based on results
    if [ $COMPLIANCE_FAILED -gt 0 ]; then
        log_error "Compliance check failed. Address failed checks before production deployment."
        exit 1
    elif [ $COMPLIANCE_WARNINGS -gt 0 ]; then
        log_warning "Compliance check completed with warnings. Review warning items."
        exit 1
    else
        log_success "All compliance checks passed successfully!"
        exit 0
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi