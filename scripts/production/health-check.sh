#!/bin/bash

# ============================================================================
# Production Health Check Script
# ============================================================================
# This script performs comprehensive health checks for the autonomous agent
# system with detailed reporting and alerting capabilities.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-30}"
CRITICAL_THRESHOLD="${CRITICAL_THRESHOLD:-90}"
WARNING_THRESHOLD="${WARNING_THRESHOLD:-80}"
REPORT_FORMAT="${REPORT_FORMAT:-text}"
NOTIFICATION_WEBHOOK="${NOTIFICATION_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
declare -A HEALTH_RESULTS
OVERALL_STATUS="healthy"
CRITICAL_ISSUES=0
WARNING_ISSUES=0

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
    -f, --format FORMAT      Output format (text, json, html)
    -t, --timeout SECONDS    Health check timeout (default: 30)
    -c, --critical PERCENT   Critical threshold (default: 90)
    -w, --warning PERCENT    Warning threshold (default: 80)
    -n, --notify URL         Notification webhook URL
    -s, --service SERVICE    Check specific service only
    -q, --quiet              Suppress output (exit code only)
    -h, --help               Show this help message

Examples:
    $0 --format json
    $0 --service postgres --timeout 60
    $0 --notify https://hooks.slack.com/services/xxx
EOF
}

# Parse command line arguments
SPECIFIC_SERVICE=""
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        -t|--timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        -c|--critical)
            CRITICAL_THRESHOLD="$2"
            shift 2
            ;;
        -w|--warning)
            WARNING_THRESHOLD="$2"
            shift 2
            ;;
        -n|--notify)
            NOTIFICATION_WEBHOOK="$2"
            shift 2
            ;;
        -s|--service)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
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

# Record health check result
record_result() {
    local service="$1"
    local status="$2"
    local message="$3"
    local response_time="${4:-0}"
    
    HEALTH_RESULTS["$service"]="$status:$message:$response_time"
    
    if [ "$status" = "critical" ]; then
        OVERALL_STATUS="critical"
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
    elif [ "$status" = "warning" ] && [ "$OVERALL_STATUS" != "critical" ]; then
        OVERALL_STATUS="warning"
        WARNING_ISSUES=$((WARNING_ISSUES + 1))
    fi
}

# HTTP health check
http_health_check() {
    local service="$1"
    local url="$2"
    local expected_status="${3:-200}"
    
    local start_time=$(date +%s.%N)
    local response_code=0
    local response_body=""
    
    # Perform HTTP request
    if response_body=$(curl -s -w "HTTPSTATUS:%{http_code}" --max-time "$HEALTH_CHECK_TIMEOUT" "$url" 2>/dev/null); then
        response_code=$(echo "$response_body" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        response_body=$(echo "$response_body" | sed 's/HTTPSTATUS:[0-9]*$//')
    else
        response_code=0
    fi
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc)
    
    # Evaluate response
    if [ "$response_code" -eq "$expected_status" ]; then
        record_result "$service" "healthy" "HTTP $response_code OK" "$response_time"
        return 0
    else
        record_result "$service" "critical" "HTTP $response_code (expected $expected_status)" "$response_time"
        return 1
    fi
}

# Database health check
database_health_check() {
    local service="$1"
    local connection_string="$2"
    
    local start_time=$(date +%s.%N)
    local result=""
    
    # Test database connection
    if result=$(docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres pg_isready -U agent 2>&1); then
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc)
        record_result "$service" "healthy" "Database connection OK" "$response_time"
        return 0
    else
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc)
        record_result "$service" "critical" "Database connection failed: $result" "$response_time"
        return 1
    fi
}

# Redis health check
redis_health_check() {
    local service="$1"
    
    local start_time=$(date +%s.%N)
    local result=""
    
    # Test Redis connection
    if result=$(docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T redis redis-cli ping 2>&1); then
        if [ "$result" = "PONG" ]; then
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc)
            record_result "$service" "healthy" "Redis connection OK" "$response_time"
            return 0
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc)
    record_result "$service" "critical" "Redis connection failed: $result" "$response_time"
    return 1
}

# Docker service health check
docker_service_health_check() {
    local service="$1"
    
    local start_time=$(date +%s.%N)
    
    # Check if service is running
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps "$service" | grep -q "Up"; then
        # Check service health status
        local health_status=$(docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps "$service" | grep "$service" | awk '{print $4}')
        
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc)
        
        if echo "$health_status" | grep -q "healthy"; then
            record_result "$service" "healthy" "Service running and healthy" "$response_time"
            return 0
        elif echo "$health_status" | grep -q "unhealthy"; then
            record_result "$service" "critical" "Service unhealthy" "$response_time"
            return 1
        else
            record_result "$service" "healthy" "Service running" "$response_time"
            return 0
        fi
    else
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc)
        record_result "$service" "critical" "Service not running" "$response_time"
        return 1
    fi
}

# System resource health check
system_health_check() {
    local service="system"
    
    local start_time=$(date +%s.%N)
    
    # Check disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    # Check memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    # Check CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local cpu_usage=$(echo "scale=0; $cpu_load * 100 / $cpu_cores" | bc)
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc)
    
    # Evaluate thresholds
    local status="healthy"
    local message="Disk: ${disk_usage}%, Memory: ${memory_usage}%, CPU: ${cpu_usage}%"
    
    if [ "$disk_usage" -gt "$CRITICAL_THRESHOLD" ] || [ "$memory_usage" -gt "$CRITICAL_THRESHOLD" ] || [ "$cpu_usage" -gt "$CRITICAL_THRESHOLD" ]; then
        status="critical"
    elif [ "$disk_usage" -gt "$WARNING_THRESHOLD" ] || [ "$memory_usage" -gt "$WARNING_THRESHOLD" ] || [ "$cpu_usage" -gt "$WARNING_THRESHOLD" ]; then
        status="warning"
    fi
    
    record_result "$service" "$status" "$message" "$response_time"
    
    [ "$status" = "healthy" ]
}

# Application-specific health checks
gmail_agent_health_check() {
    http_health_check "gmail-agent" "http://localhost:8000/api/v1/agents/gmail/health"
}

research_agent_health_check() {
    http_health_check "research-agent" "http://localhost:8000/api/v1/agents/research/health"
}

code_agent_health_check() {
    http_health_check "code-agent" "http://localhost:8000/api/v1/agents/code/health"
}

intelligence_engine_health_check() {
    http_health_check "intelligence-engine" "http://localhost:8000/api/v1/intelligence/health"
}

api_health_check() {
    http_health_check "api" "http://localhost:8000/api/v1/health"
}

main_app_health_check() {
    http_health_check "main-app" "http://localhost:8000/health"
}

# Ollama health check
ollama_health_check() {
    http_health_check "ollama" "http://localhost:11434/api/tags"
}

# Monitoring services health check
prometheus_health_check() {
    http_health_check "prometheus" "http://localhost:9090/-/healthy"
}

grafana_health_check() {
    http_health_check "grafana" "http://localhost:3000/api/health"
}

# Send notification
send_notification() {
    local webhook_url="$1"
    local status="$2"
    local message="$3"
    
    if [ -z "$webhook_url" ]; then
        return 0
    fi
    
    local color="good"
    if [ "$status" = "critical" ]; then
        color="danger"
    elif [ "$status" = "warning" ]; then
        color="warning"
    fi
    
    local payload=$(cat <<EOF
{
    "text": "Autonomous Agent Health Check",
    "attachments": [
        {
            "color": "$color",
            "title": "Health Check Status: $status",
            "text": "$message",
            "timestamp": $(date +%s)
        }
    ]
}
EOF
)
    
    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$webhook_url" > /dev/null || true
}

# Generate text report
generate_text_report() {
    echo "=================================="
    echo "Autonomous Agent Health Check Report"
    echo "=================================="
    echo "Timestamp: $(date)"
    echo "Overall Status: $OVERALL_STATUS"
    echo "Critical Issues: $CRITICAL_ISSUES"
    echo "Warning Issues: $WARNING_ISSUES"
    echo ""
    
    echo "Service Health Details:"
    echo "----------------------"
    printf "%-20s %-10s %-15s %-s\n" "Service" "Status" "Response Time" "Message"
    printf "%-20s %-10s %-15s %-s\n" "-------" "------" "-------------" "-------"
    
    for service in "${!HEALTH_RESULTS[@]}"; do
        IFS=':' read -r status message response_time <<< "${HEALTH_RESULTS[$service]}"
        printf "%-20s %-10s %-15s %-s\n" "$service" "$status" "${response_time}s" "$message"
    done
    
    echo ""
}

# Generate JSON report
generate_json_report() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    echo "{"
    echo "  \"timestamp\": \"$timestamp\","
    echo "  \"overall_status\": \"$OVERALL_STATUS\","
    echo "  \"critical_issues\": $CRITICAL_ISSUES,"
    echo "  \"warning_issues\": $WARNING_ISSUES,"
    echo "  \"services\": {"
    
    local first=true
    for service in "${!HEALTH_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        
        IFS=':' read -r status message response_time <<< "${HEALTH_RESULTS[$service]}"
        echo "    \"$service\": {"
        echo "      \"status\": \"$status\","
        echo "      \"message\": \"$message\","
        echo "      \"response_time\": $response_time"
        echo -n "    }"
    done
    
    echo ""
    echo "  }"
    echo "}"
}

# Generate HTML report
generate_html_report() {
    local timestamp=$(date)
    
    cat << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Autonomous Agent Health Check Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .healthy { color: green; }
        .warning { color: orange; }
        .critical { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Autonomous Agent Health Check Report</h1>
    <p><strong>Timestamp:</strong> $timestamp</p>
    <p><strong>Overall Status:</strong> <span class="$OVERALL_STATUS">$OVERALL_STATUS</span></p>
    <p><strong>Critical Issues:</strong> $CRITICAL_ISSUES</p>
    <p><strong>Warning Issues:</strong> $WARNING_ISSUES</p>
    
    <h2>Service Health Details</h2>
    <table>
        <tr>
            <th>Service</th>
            <th>Status</th>
            <th>Response Time</th>
            <th>Message</th>
        </tr>
EOF
    
    for service in "${!HEALTH_RESULTS[@]}"; do
        IFS=':' read -r status message response_time <<< "${HEALTH_RESULTS[$service]}"
        echo "        <tr>"
        echo "            <td>$service</td>"
        echo "            <td class=\"$status\">$status</td>"
        echo "            <td>${response_time}s</td>"
        echo "            <td>$message</td>"
        echo "        </tr>"
    done
    
    cat << EOF
    </table>
</body>
</html>
EOF
}

# Main health check function
main() {
    if [ "$QUIET" = false ]; then
        log_info "Starting autonomous agent health check"
    fi
    
    cd "$PROJECT_ROOT"
    
    # Define health check functions
    declare -A HEALTH_CHECKS=(
        ["system"]="system_health_check"
        ["postgres"]="database_health_check postgres"
        ["redis"]="redis_health_check redis"
        ["ollama"]="ollama_health_check"
        ["app"]="docker_service_health_check app"
        ["nginx"]="docker_service_health_check nginx"
        ["main-app"]="main_app_health_check"
        ["api"]="api_health_check"
        ["gmail-agent"]="gmail_agent_health_check"
        ["research-agent"]="research_agent_health_check"
        ["code-agent"]="code_agent_health_check"
        ["intelligence-engine"]="intelligence_engine_health_check"
        ["prometheus"]="prometheus_health_check"
        ["grafana"]="grafana_health_check"
    )
    
    # Run health checks
    if [ -n "$SPECIFIC_SERVICE" ]; then
        if [ -n "${HEALTH_CHECKS[$SPECIFIC_SERVICE]}" ]; then
            ${HEALTH_CHECKS[$SPECIFIC_SERVICE]}
        else
            log_error "Unknown service: $SPECIFIC_SERVICE"
            exit 1
        fi
    else
        for service in "${!HEALTH_CHECKS[@]}"; do
            if [ "$QUIET" = false ]; then
                log_info "Checking $service..."
            fi
            ${HEALTH_CHECKS[$service]} || true
        done
    fi
    
    # Generate report
    if [ "$QUIET" = false ]; then
        case "$REPORT_FORMAT" in
            "json")
                generate_json_report
                ;;
            "html")
                generate_html_report
                ;;
            *)
                generate_text_report
                ;;
        esac
    fi
    
    # Send notification if configured
    if [ -n "$NOTIFICATION_WEBHOOK" ] && [ "$OVERALL_STATUS" != "healthy" ]; then
        local message="Health check completed with $OVERALL_STATUS status. Critical: $CRITICAL_ISSUES, Warnings: $WARNING_ISSUES"
        send_notification "$NOTIFICATION_WEBHOOK" "$OVERALL_STATUS" "$message"
    fi
    
    # Set exit code based on overall status
    case "$OVERALL_STATUS" in
        "healthy")
            exit 0
            ;;
        "warning")
            exit 1
            ;;
        "critical")
            exit 2
            ;;
    esac
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi