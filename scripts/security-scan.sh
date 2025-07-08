#!/bin/bash

# ============================================================================
# Security Scanning Script for Autonomous Agent System
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCAN_DIR="./security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$SCAN_DIR/$TIMESTAMP"

# Create report directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}Autonomous Agent System - Security Scanner${NC}"
echo -e "============================================="
echo -e "Report directory: $REPORT_DIR"
echo ""

# Function to run Python security scans
run_python_security_scans() {
    echo -e "${YELLOW}Running Python security scans...${NC}"
    
    # Bandit - Python security scanner
    if command -v bandit &> /dev/null; then
        echo "Running Bandit security scan..."
        bandit -r src/ -f json -o "$REPORT_DIR/bandit-report.json" || true
        bandit -r src/ -f txt -o "$REPORT_DIR/bandit-report.txt" || true
        echo -e "${GREEN}Bandit scan completed${NC}"
    else
        echo -e "${RED}Bandit not found, installing...${NC}"
        pip install bandit
        bandit -r src/ -f json -o "$REPORT_DIR/bandit-report.json" || true
        bandit -r src/ -f txt -o "$REPORT_DIR/bandit-report.txt" || true
    fi
    
    # Safety - Python dependency vulnerability scanner
    if command -v safety &> /dev/null; then
        echo "Running Safety dependency scan..."
        safety check --json --output "$REPORT_DIR/safety-report.json" || true
        safety check --output "$REPORT_DIR/safety-report.txt" || true
        echo -e "${GREEN}Safety scan completed${NC}"
    else
        echo -e "${RED}Safety not found, installing...${NC}"
        pip install safety
        safety check --json --output "$REPORT_DIR/safety-report.json" || true
        safety check --output "$REPORT_DIR/safety-report.txt" || true
    fi
    
    # Semgrep - Static analysis
    if command -v semgrep &> /dev/null; then
        echo "Running Semgrep static analysis..."
        semgrep --config=auto src/ --json --output="$REPORT_DIR/semgrep-report.json" || true
        semgrep --config=auto src/ --output="$REPORT_DIR/semgrep-report.txt" || true
        echo -e "${GREEN}Semgrep scan completed${NC}"
    else
        echo -e "${YELLOW}Semgrep not found, skipping...${NC}"
    fi
}

# Function to run Docker security scans
run_docker_security_scans() {
    echo -e "${YELLOW}Running Docker security scans...${NC}"
    
    # Docker security scan with Docker Scout (if available)
    if command -v docker &> /dev/null; then
        echo "Running Docker image security scan..."
        
        # Build image for scanning
        docker build -t autonomous-agent:security-scan . || true
        
        # Docker Scout scan (if available)
        if docker scout version &> /dev/null; then
            docker scout cves autonomous-agent:security-scan --format json --output "$REPORT_DIR/docker-scout-report.json" || true
            docker scout cves autonomous-agent:security-scan --output "$REPORT_DIR/docker-scout-report.txt" || true
            echo -e "${GREEN}Docker Scout scan completed${NC}"
        else
            echo -e "${YELLOW}Docker Scout not available${NC}"
        fi
        
        # Trivy scan (if available)
        if command -v trivy &> /dev/null; then
            trivy image --format json --output "$REPORT_DIR/trivy-report.json" autonomous-agent:security-scan || true
            trivy image --format table --output "$REPORT_DIR/trivy-report.txt" autonomous-agent:security-scan || true
            echo -e "${GREEN}Trivy scan completed${NC}"
        else
            echo -e "${YELLOW}Trivy not found, installing...${NC}"
            # Install Trivy (Linux)
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
                trivy image --format json --output "$REPORT_DIR/trivy-report.json" autonomous-agent:security-scan || true
                trivy image --format table --output "$REPORT_DIR/trivy-report.txt" autonomous-agent:security-scan || true
            else
                echo -e "${YELLOW}Trivy installation not supported on this platform${NC}"
            fi
        fi
    else
        echo -e "${RED}Docker not available${NC}"
    fi
}

# Function to run secrets scanning
run_secrets_scan() {
    echo -e "${YELLOW}Running secrets scanning...${NC}"
    
    # GitLeaks (if available)
    if command -v gitleaks &> /dev/null; then
        echo "Running GitLeaks secrets scan..."
        gitleaks detect --source . --report-format json --report-path "$REPORT_DIR/gitleaks-report.json" || true
        gitleaks detect --source . --report-format csv --report-path "$REPORT_DIR/gitleaks-report.csv" || true
        echo -e "${GREEN}GitLeaks scan completed${NC}"
    else
        echo -e "${YELLOW}GitLeaks not found, installing...${NC}"
        # Install GitLeaks
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -sSfL https://github.com/zricethezav/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz | tar -xz -C /usr/local/bin/
            gitleaks detect --source . --report-format json --report-path "$REPORT_DIR/gitleaks-report.json" || true
        else
            echo -e "${YELLOW}GitLeaks installation not supported on this platform${NC}"
        fi
    fi
    
    # TruffleHog (if available)
    if command -v trufflehog &> /dev/null; then
        echo "Running TruffleHog secrets scan..."
        trufflehog filesystem . --json > "$REPORT_DIR/trufflehog-report.json" || true
        echo -e "${GREEN}TruffleHog scan completed${NC}"
    else
        echo -e "${YELLOW}TruffleHog not found${NC}"
    fi
}

# Function to run Kubernetes security scans
run_kubernetes_security_scans() {
    echo -e "${YELLOW}Running Kubernetes security scans...${NC}"
    
    # Kube-score (if available)
    if command -v kube-score &> /dev/null; then
        echo "Running kube-score security scan..."
        kube-score score k8s/base/*.yaml > "$REPORT_DIR/kube-score-report.txt" || true
        echo -e "${GREEN}Kube-score scan completed${NC}"
    else
        echo -e "${YELLOW}Kube-score not found${NC}"
    fi
    
    # Kube-bench (if available)
    if command -v kube-bench &> /dev/null; then
        echo "Running kube-bench security scan..."
        kube-bench --json > "$REPORT_DIR/kube-bench-report.json" || true
        echo -e "${GREEN}Kube-bench scan completed${NC}"
    else
        echo -e "${YELLOW}Kube-bench not found${NC}"
    fi
    
    # Polaris (if available)
    if command -v polaris &> /dev/null; then
        echo "Running Polaris security scan..."
        polaris audit --audit-path k8s/ --format json > "$REPORT_DIR/polaris-report.json" || true
        echo -e "${GREEN}Polaris scan completed${NC}"
    else
        echo -e "${YELLOW}Polaris not found${NC}"
    fi
}

# Function to run dependency vulnerability scans
run_dependency_scans() {
    echo -e "${YELLOW}Running dependency vulnerability scans...${NC}"
    
    # npm audit (if package.json exists)
    if [ -f "package.json" ]; then
        echo "Running npm audit..."
        npm audit --json > "$REPORT_DIR/npm-audit-report.json" || true
        npm audit > "$REPORT_DIR/npm-audit-report.txt" || true
        echo -e "${GREEN}npm audit completed${NC}"
    fi
    
    # pip-audit (if requirements.txt exists)
    if [ -f "requirements.txt" ]; then
        echo "Running pip-audit..."
        if command -v pip-audit &> /dev/null; then
            pip-audit --format=json --output="$REPORT_DIR/pip-audit-report.json" || true
            pip-audit --output="$REPORT_DIR/pip-audit-report.txt" || true
            echo -e "${GREEN}pip-audit completed${NC}"
        else
            echo -e "${YELLOW}pip-audit not found, installing...${NC}"
            pip install pip-audit
            pip-audit --format=json --output="$REPORT_DIR/pip-audit-report.json" || true
        fi
    fi
}

# Function to generate consolidated report
generate_consolidated_report() {
    echo -e "${YELLOW}Generating consolidated security report...${NC}"
    
    cat > "$REPORT_DIR/security-summary.html" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .critical { color: #d32f2f; }
        .warning { color: #f57c00; }
        .info { color: #1976d2; }
        .success { color: #388e3c; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p>Generated: $(date)</p>
        <p>Timestamp: $TIMESTAMP</p>
    </div>
    
    <div class="section">
        <h2>Scan Summary</h2>
        <table>
            <tr><th>Scan Type</th><th>Status</th><th>Report File</th></tr>
EOF

    # Add scan results to HTML report
    for report_file in "$REPORT_DIR"/*.json "$REPORT_DIR"/*.txt; do
        if [ -f "$report_file" ]; then
            filename=$(basename "$report_file")
            echo "<tr><td>$filename</td><td class=\"success\">Completed</td><td>$filename</td></tr>" >> "$REPORT_DIR/security-summary.html"
        fi
    done
    
    cat >> "$REPORT_DIR/security-summary.html" <<EOF
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>Review all critical and high-severity vulnerabilities</li>
            <li>Update dependencies with known vulnerabilities</li>
            <li>Implement recommended security configurations</li>
            <li>Regular security scanning as part of CI/CD pipeline</li>
            <li>Monitor security advisories for used dependencies</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Files in this Report</h2>
        <ul>
EOF

    for report_file in "$REPORT_DIR"/*; do
        if [ -f "$report_file" ]; then
            filename=$(basename "$report_file")
            echo "<li><a href=\"$filename\">$filename</a></li>" >> "$REPORT_DIR/security-summary.html"
        fi
    done
    
    cat >> "$REPORT_DIR/security-summary.html" <<EOF
        </ul>
    </div>
</body>
</html>
EOF

    echo -e "${GREEN}Consolidated report generated: $REPORT_DIR/security-summary.html${NC}"
}

# Function to show scan results summary
show_scan_summary() {
    echo ""
    echo -e "${BLUE}Security Scan Summary${NC}"
    echo -e "====================="
    echo -e "Scan completed: $(date)"
    echo -e "Reports directory: $REPORT_DIR"
    echo ""
    
    # Count report files
    report_count=$(find "$REPORT_DIR" -name "*.json" -o -name "*.txt" | wc -l)
    echo -e "Generated reports: $report_count"
    
    # List report files
    echo -e "\nReport files:"
    for report_file in "$REPORT_DIR"/*; do
        if [ -f "$report_file" ]; then
            filename=$(basename "$report_file")
            size=$(du -h "$report_file" | cut -f1)
            echo -e "  - $filename ($size)"
        fi
    done
    
    echo ""
    echo -e "${GREEN}Security scan completed successfully!${NC}"
    echo -e "Review the reports in: $REPORT_DIR"
    echo -e "Open security-summary.html in a browser for consolidated view"
}

# Main execution
main() {
    echo -e "${BLUE}Starting comprehensive security scan...${NC}"
    
    # Run all security scans
    run_python_security_scans
    run_docker_security_scans
    run_secrets_scan
    run_kubernetes_security_scans
    run_dependency_scans
    
    # Generate consolidated report
    generate_consolidated_report
    
    # Show summary
    show_scan_summary
}

# Handle command line arguments
case "${1:-}" in
    "python")
        run_python_security_scans
        ;;
    "docker")
        run_docker_security_scans
        ;;
    "secrets")
        run_secrets_scan
        ;;
    "kubernetes")
        run_kubernetes_security_scans
        ;;
    "dependencies")
        run_dependency_scans
        ;;
    "all"|"")
        main
        ;;
    *)
        echo "Usage: $0 {python|docker|secrets|kubernetes|dependencies|all}"
        echo "  python       - Run Python security scans"
        echo "  docker       - Run Docker security scans"
        echo "  secrets      - Run secrets scanning"
        echo "  kubernetes   - Run Kubernetes security scans"
        echo "  dependencies - Run dependency vulnerability scans"
        echo "  all          - Run all security scans (default)"
        exit 1
        ;;
esac