#!/bin/bash

# ============================================================================
# Production Testing and Validation Script
# ============================================================================
# This script performs comprehensive testing and validation of the autonomous
# agent system in production environment.
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_RESULTS_DIR="${PROJECT_ROOT}/test-results"
LOAD_TEST_DURATION="${LOAD_TEST_DURATION:-300}"
CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

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
    -u, --base-url URL       Base URL for testing (default: http://localhost:8000)
    -k, --api-key KEY        API key for authentication
    -d, --duration SECONDS   Load test duration (default: 300)
    -c, --concurrent USERS   Concurrent users for load testing (default: 50)
    -t, --test-type TYPE     Test type (all, unit, integration, load, security)
    -r, --results-dir DIR    Test results directory
    -v, --verbose            Verbose output
    -h, --help               Show this help message

Test Types:
    all          - Run all tests
    unit         - Run unit tests
    integration  - Run integration tests
    load         - Run load tests
    security     - Run security tests
    smoke        - Run smoke tests
    e2e          - Run end-to-end tests

Examples:
    $0 --test-type all
    $0 --test-type load --duration 600 --concurrent 100
    $0 --test-type security --base-url https://prod.example.com
EOF
}

# Parse command line arguments
TEST_TYPE="all"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--base-url)
            BASE_URL="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -d|--duration)
            LOAD_TEST_DURATION="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        -t|--test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -r|--results-dir)
            TEST_RESULTS_DIR="$2"
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

# Initialize test environment
initialize_testing() {
    log_info "Initializing test environment..."
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Create test report
    cat > "$TEST_RESULTS_DIR/test-report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Production Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .passed { color: green; }
        .failed { color: red; }
        .skipped { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Production Test Report</h1>
    <p>Generated: $(date)</p>
    <p>Base URL: $BASE_URL</p>
    <p>Test Type: $TEST_TYPE</p>
    <div id="results">
EOF
    
    # Install test dependencies if not present
    if ! command -v curl &> /dev/null; then
        log_error "curl is required for testing"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found, installing..."
        sudo apt-get update && sudo apt-get install -y jq
    fi
    
    log_success "Test environment initialized"
}

# Test result tracking
record_test_result() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    local duration="${4:-0}"
    
    case "$status" in
        "PASSED")
            TESTS_PASSED=$((TESTS_PASSED + 1))
            log_success "$test_name: $message"
            ;;
        "FAILED")
            TESTS_FAILED=$((TESTS_FAILED + 1))
            log_error "$test_name: $message"
            ;;
        "SKIPPED")
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
            log_warning "$test_name: $message"
            ;;
    esac
    
    # Add to HTML report
    cat >> "$TEST_RESULTS_DIR/test-report.html" << EOF
    <div class="test-result">
        <h3 class="$status">$test_name</h3>
        <p>Status: <span class="$(echo $status | tr '[:upper:]' '[:lower:]')">$status</span></p>
        <p>Message: $message</p>
        <p>Duration: ${duration}s</p>
        <hr>
    </div>
EOF
}

# HTTP request helper
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"
    local expected_status="${4:-200}"
    
    local auth_header=""
    if [ -n "$API_KEY" ]; then
        auth_header="-H \"Authorization: Bearer $API_KEY\""
    fi
    
    local curl_cmd="curl -s -w \"%{http_code}\" -X $method"
    if [ -n "$data" ]; then
        curl_cmd="$curl_cmd -H \"Content-Type: application/json\" -d '$data'"
    fi
    if [ -n "$auth_header" ]; then
        curl_cmd="$curl_cmd $auth_header"
    fi
    curl_cmd="$curl_cmd \"$BASE_URL$endpoint\""
    
    local response=$(eval $curl_cmd)
    local status_code="${response: -3}"
    local body="${response%???}"
    
    if [ "$status_code" -eq "$expected_status" ]; then
        echo "$body"
        return 0
    else
        echo "HTTP $status_code: $body"
        return 1
    fi
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test 1: Health check
    local start_time=$(date +%s.%N)
    if response=$(make_request "GET" "/health" "" 200); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Health Check" "PASSED" "System is healthy" "$duration"
    else
        record_test_result "Health Check" "FAILED" "Health check failed: $response"
    fi
    
    # Test 2: API status
    local start_time=$(date +%s.%N)
    if response=$(make_request "GET" "/api/v1/status" "" 200); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "API Status" "PASSED" "API is responding" "$duration"
    else
        record_test_result "API Status" "FAILED" "API status check failed: $response"
    fi
    
    # Test 3: Agent health checks
    local agents=("gmail" "research" "code" "intelligence")
    for agent in "${agents[@]}"; do
        local start_time=$(date +%s.%N)
        if response=$(make_request "GET" "/api/v1/agents/$agent/health" "" 200); then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            record_test_result "$agent Agent Health" "PASSED" "Agent is healthy" "$duration"
        else
            record_test_result "$agent Agent Health" "FAILED" "Agent health check failed: $response"
        fi
    done
}

# Unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run Python unit tests
    if python -m pytest tests/unit/ -v --tb=short --junit-xml="$TEST_RESULTS_DIR/unit-tests.xml" 2>&1 | tee "$TEST_RESULTS_DIR/unit-tests.log"; then
        record_test_result "Unit Tests" "PASSED" "All unit tests passed"
    else
        record_test_result "Unit Tests" "FAILED" "Some unit tests failed"
    fi
    
    # Run code quality checks
    if python -m ruff check src/ tests/ 2>&1 | tee "$TEST_RESULTS_DIR/ruff-check.log"; then
        record_test_result "Code Quality" "PASSED" "Code quality checks passed"
    else
        record_test_result "Code Quality" "FAILED" "Code quality issues found"
    fi
    
    # Run type checking
    if python -m mypy src/ 2>&1 | tee "$TEST_RESULTS_DIR/mypy-check.log"; then
        record_test_result "Type Checking" "PASSED" "Type checking passed"
    else
        record_test_result "Type Checking" "FAILED" "Type checking issues found"
    fi
}

# Integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Test database connectivity
    local start_time=$(date +%s.%N)
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres pg_isready -U agent; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Database Connectivity" "PASSED" "Database is accessible" "$duration"
    else
        record_test_result "Database Connectivity" "FAILED" "Cannot connect to database"
    fi
    
    # Test Redis connectivity
    local start_time=$(date +%s.%N)
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T redis redis-cli ping | grep -q "PONG"; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Redis Connectivity" "PASSED" "Redis is accessible" "$duration"
    else
        record_test_result "Redis Connectivity" "FAILED" "Cannot connect to Redis"
    fi
    
    # Test Ollama connectivity
    local start_time=$(date +%s.%N)
    if response=$(make_request "GET" "/api/v1/ollama/models" "" 200); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Ollama Connectivity" "PASSED" "Ollama is accessible" "$duration"
    else
        record_test_result "Ollama Connectivity" "FAILED" "Cannot connect to Ollama"
    fi
    
    # Test agent integration
    if [ -n "$API_KEY" ]; then
        # Test Gmail agent integration
        local start_time=$(date +%s.%N)
        local test_data='{"query": "test", "limit": 1}'
        if response=$(make_request "POST" "/api/v1/agents/gmail/search" "$test_data" 200); then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            record_test_result "Gmail Integration" "PASSED" "Gmail agent integration works" "$duration"
        else
            record_test_result "Gmail Integration" "FAILED" "Gmail agent integration failed"
        fi
        
        # Test Research agent integration
        local start_time=$(date +%s.%N)
        local test_data='{"query": "test search", "limit": 1}'
        if response=$(make_request "POST" "/api/v1/agents/research/search" "$test_data" 200); then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            record_test_result "Research Integration" "PASSED" "Research agent integration works" "$duration"
        else
            record_test_result "Research Integration" "FAILED" "Research agent integration failed"
        fi
    else
        record_test_result "Agent Integration" "SKIPPED" "No API key provided"
    fi
}

# Load tests
run_load_tests() {
    log_info "Running load tests..."
    
    # Check if Apache Bench is available
    if ! command -v ab &> /dev/null; then
        log_warning "Apache Bench not found, installing..."
        sudo apt-get update && sudo apt-get install -y apache2-utils
    fi
    
    # Test 1: Health endpoint load test
    log_info "Load testing health endpoint..."
    local start_time=$(date +%s.%N)
    if ab -n 1000 -c 10 -g "$TEST_RESULTS_DIR/health-load.data" "$BASE_URL/health" > "$TEST_RESULTS_DIR/health-load.log" 2>&1; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        local avg_response_time=$(grep "Time per request" "$TEST_RESULTS_DIR/health-load.log" | head -1 | awk '{print $4}')
        record_test_result "Health Load Test" "PASSED" "Average response time: ${avg_response_time}ms" "$duration"
    else
        record_test_result "Health Load Test" "FAILED" "Load test failed"
    fi
    
    # Test 2: API endpoint load test
    if [ -n "$API_KEY" ]; then
        log_info "Load testing API endpoint..."
        local start_time=$(date +%s.%N)
        if ab -n 500 -c 5 -H "Authorization: Bearer $API_KEY" -g "$TEST_RESULTS_DIR/api-load.data" "$BASE_URL/api/v1/status" > "$TEST_RESULTS_DIR/api-load.log" 2>&1; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            local avg_response_time=$(grep "Time per request" "$TEST_RESULTS_DIR/api-load.log" | head -1 | awk '{print $4}')
            record_test_result "API Load Test" "PASSED" "Average response time: ${avg_response_time}ms" "$duration"
        else
            record_test_result "API Load Test" "FAILED" "API load test failed"
        fi
    else
        record_test_result "API Load Test" "SKIPPED" "No API key provided"
    fi
    
    # Test 3: Concurrent user simulation
    log_info "Simulating concurrent users..."
    local start_time=$(date +%s.%N)
    
    # Create concurrent user test script
    cat > "$TEST_RESULTS_DIR/concurrent-test.sh" << 'EOF'
#!/bin/bash
for i in {1..10}; do
    curl -s "$1/health" > /dev/null &
    curl -s "$1/api/v1/status" > /dev/null &
done
wait
EOF
    
    chmod +x "$TEST_RESULTS_DIR/concurrent-test.sh"
    
    # Run concurrent test
    if timeout 60 "$TEST_RESULTS_DIR/concurrent-test.sh" "$BASE_URL"; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Concurrent Users" "PASSED" "Concurrent user simulation completed" "$duration"
    else
        record_test_result "Concurrent Users" "FAILED" "Concurrent user simulation failed or timed out"
    fi
}

# Security tests
run_security_tests() {
    log_info "Running security tests..."
    
    # Test 1: SSL/TLS configuration
    if [[ "$BASE_URL" == https://* ]]; then
        local start_time=$(date +%s.%N)
        if openssl s_client -connect "$(echo "$BASE_URL" | sed 's|https://||'):443" -servername "$(echo "$BASE_URL" | sed 's|https://||')" < /dev/null 2>&1 | grep -q "Verify return code: 0"; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            record_test_result "SSL/TLS Configuration" "PASSED" "SSL certificate is valid" "$duration"
        else
            record_test_result "SSL/TLS Configuration" "FAILED" "SSL certificate validation failed"
        fi
    else
        record_test_result "SSL/TLS Configuration" "SKIPPED" "HTTP endpoint (SSL not applicable)"
    fi
    
    # Test 2: Authentication bypass attempt
    local start_time=$(date +%s.%N)
    if response=$(make_request "GET" "/api/v1/admin/users" "" 401); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Authentication Bypass" "PASSED" "Authentication is properly enforced" "$duration"
    else
        record_test_result "Authentication Bypass" "FAILED" "Authentication bypass possible"
    fi
    
    # Test 3: SQL injection attempt
    local start_time=$(date +%s.%N)
    local malicious_data='{"query": "test\"; DROP TABLE users; --", "limit": 1}'
    if response=$(make_request "POST" "/api/v1/agents/research/search" "$malicious_data" 400); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "SQL Injection Protection" "PASSED" "SQL injection attempt blocked" "$duration"
    else
        record_test_result "SQL Injection Protection" "FAILED" "SQL injection protection may be insufficient"
    fi
    
    # Test 4: XSS protection
    local start_time=$(date +%s.%N)
    local xss_data='{"query": "<script>alert(\"XSS\")</script>", "limit": 1}'
    if response=$(make_request "POST" "/api/v1/agents/research/search" "$xss_data" 400); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "XSS Protection" "PASSED" "XSS attempt blocked" "$duration"
    else
        record_test_result "XSS Protection" "FAILED" "XSS protection may be insufficient"
    fi
    
    # Test 5: Rate limiting
    local start_time=$(date +%s.%N)
    local rate_limit_failed=false
    for i in {1..100}; do
        if ! make_request "GET" "/health" "" 200 > /dev/null 2>&1; then
            rate_limit_failed=true
            break
        fi
    done
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if [ "$rate_limit_failed" = true ]; then
        record_test_result "Rate Limiting" "PASSED" "Rate limiting is active" "$duration"
    else
        record_test_result "Rate Limiting" "WARNING" "Rate limiting may be too permissive" "$duration"
    fi
}

# End-to-end tests
run_e2e_tests() {
    log_info "Running end-to-end tests..."
    
    if [ -z "$API_KEY" ]; then
        record_test_result "E2E Tests" "SKIPPED" "No API key provided for E2E tests"
        return
    fi
    
    # Test 1: Complete workflow test
    local start_time=$(date +%s.%N)
    
    # Create a task
    local task_data='{"title": "Test Task", "description": "E2E test task", "priority": "low"}'
    if task_response=$(make_request "POST" "/api/v1/tasks" "$task_data" 201); then
        local task_id=$(echo "$task_response" | jq -r '.task_id')
        
        # Get task details
        if task_details=$(make_request "GET" "/api/v1/tasks/$task_id" "" 200); then
            # Update task
            local update_data='{"status": "completed"}'
            if make_request "PUT" "/api/v1/tasks/$task_id" "$update_data" 200 > /dev/null; then
                local end_time=$(date +%s.%N)
                local duration=$(echo "$end_time - $start_time" | bc)
                record_test_result "E2E Workflow" "PASSED" "Complete workflow test successful" "$duration"
            else
                record_test_result "E2E Workflow" "FAILED" "Task update failed"
            fi
        else
            record_test_result "E2E Workflow" "FAILED" "Task retrieval failed"
        fi
    else
        record_test_result "E2E Workflow" "FAILED" "Task creation failed"
    fi
    
    # Test 2: Multi-agent coordination
    local start_time=$(date +%s.%N)
    local coordination_data='{"task": {"description": "Test coordination", "priority": "medium"}, "agents": ["research", "gmail"]}'
    if response=$(make_request "POST" "/api/v1/intelligence/coordinate" "$coordination_data" 200); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        record_test_result "Multi-Agent Coordination" "PASSED" "Agent coordination successful" "$duration"
    else
        record_test_result "Multi-Agent Coordination" "FAILED" "Agent coordination failed"
    fi
}

# Performance validation
run_performance_tests() {
    log_info "Running performance validation..."
    
    # Test response time thresholds
    local start_time=$(date +%s.%N)
    if response=$(make_request "GET" "/health" "" 200); then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        # Check if response time is under 1 second
        if (( $(echo "$duration < 1.0" | bc -l) )); then
            record_test_result "Response Time" "PASSED" "Response time: ${duration}s (< 1s)" "$duration"
        else
            record_test_result "Response Time" "FAILED" "Response time: ${duration}s (> 1s)" "$duration"
        fi
    else
        record_test_result "Response Time" "FAILED" "Health check failed"
    fi
    
    # Test memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage < 80.0" | bc -l) )); then
        record_test_result "Memory Usage" "PASSED" "Memory usage: ${memory_usage}% (< 80%)"
    else
        record_test_result "Memory Usage" "WARNING" "Memory usage: ${memory_usage}% (> 80%)"
    fi
    
    # Test disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        record_test_result "Disk Usage" "PASSED" "Disk usage: ${disk_usage}% (< 80%)"
    else
        record_test_result "Disk Usage" "WARNING" "Disk usage: ${disk_usage}% (> 80%)"
    fi
}

# Generate final report
generate_report() {
    log_info "Generating test report..."
    
    # Close HTML report
    cat >> "$TEST_RESULTS_DIR/test-report.html" << EOF
    </div>
    <h2>Test Summary</h2>
    <table>
        <tr><th>Status</th><th>Count</th></tr>
        <tr><td class="passed">Passed</td><td>$TESTS_PASSED</td></tr>
        <tr><td class="failed">Failed</td><td>$TESTS_FAILED</td></tr>
        <tr><td class="skipped">Skipped</td><td>$TESTS_SKIPPED</td></tr>
    </table>
    <p>Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))</p>
    <p>Success Rate: $(( TESTS_PASSED * 100 / (TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED) ))%</p>
</body>
</html>
EOF
    
    # Generate summary
    cat > "$TEST_RESULTS_DIR/summary.txt" << EOF
Production Test Summary
======================
Date: $(date)
Base URL: $BASE_URL
Test Type: $TEST_TYPE

Results:
- Passed: $TESTS_PASSED
- Failed: $TESTS_FAILED
- Skipped: $TESTS_SKIPPED
- Total: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

Success Rate: $(( TESTS_PASSED * 100 / (TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED) ))%
EOF
    
    # Generate JUnit XML report
    cat > "$TEST_RESULTS_DIR/junit-report.xml" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="ProductionTests" tests="$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))" failures="$TESTS_FAILED" skipped="$TESTS_SKIPPED" time="$(date +%s)">
        <!-- Individual test results would go here -->
    </testsuite>
</testsuites>
EOF
    
    log_info "Test report generated: $TEST_RESULTS_DIR/test-report.html"
    log_info "Test summary: $TEST_RESULTS_DIR/summary.txt"
    
    # Print summary
    echo
    echo "=================================="
    echo "Production Test Summary"
    echo "=================================="
    echo "Passed:  $TESTS_PASSED"
    echo "Failed:  $TESTS_FAILED"
    echo "Skipped: $TESTS_SKIPPED"
    echo "Total:   $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
    echo "Success Rate: $(( TESTS_PASSED * 100 / (TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED) ))%"
    echo "=================================="
}

# Main function
main() {
    log_info "Starting production testing and validation"
    log_info "Test type: $TEST_TYPE"
    log_info "Base URL: $BASE_URL"
    log_info "Results directory: $TEST_RESULTS_DIR"
    
    # Initialize testing environment
    initialize_testing
    
    # Run tests based on type
    case "$TEST_TYPE" in
        "all")
            run_smoke_tests
            run_unit_tests
            run_integration_tests
            run_load_tests
            run_security_tests
            run_e2e_tests
            run_performance_tests
            ;;
        "smoke")
            run_smoke_tests
            ;;
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "load")
            run_load_tests
            ;;
        "security")
            run_security_tests
            ;;
        "e2e")
            run_e2e_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac
    
    # Generate final report
    generate_report
    
    # Exit with appropriate code
    if [ $TESTS_FAILED -gt 0 ]; then
        log_error "Some tests failed. Check the test report for details."
        exit 1
    else
        log_success "All tests passed successfully!"
        exit 0
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi