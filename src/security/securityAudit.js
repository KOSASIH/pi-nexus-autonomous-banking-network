// securityAudit.js

const fs = require('fs');
const path = require('path');

class SecurityAudit {
    constructor () {
        this.vulnerabilities = [];
    }

    // Scan a file for common security issues
    scanFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf-8');
        this.checkForVulnerabilities(content, filePath);
    }

    // Check for specific vulnerabilities in the code
    checkForVulnerabilities(content, filePath) {
        const patterns = {
            'eval': /eval\(/g,
            'exec': /exec\(/g,
            'unhandledPromiseRejection': /process\.on\('unhandledRejection'/g,
            'insecureRandom': /Math\.random\(\)/g,
            'httpRequests': /http\.get\(/g,
        };

        for (const [key, pattern] of Object.entries(patterns)) {
            const matches = content.match(pattern);
            if (matches) {
                this.vulnerabilities.push({
                    type: key,
                    file: filePath,
                    occurrences: matches.length,
                });
            }
        }
    }

    // Recursively scan a directory for files
    scanDirectory(directoryPath) {
        const files = fs.readdirSync(directoryPath);
        for (const file of files) {
            const fullPath = path.join(directoryPath, file);
            if (fs.statSync(fullPath).isDirectory()) {
                this.scanDirectory(fullPath);
            } else if (fullPath.endsWith('.js')) {
                this.scanFile(fullPath);
            }
        }
    }

    // Generate a report of the vulnerabilities found
    generateReport() {
        if (this.vulnerabilities.length === 0) {
            console.log('No vulnerabilities found.');
            return;
        }

        console.log('Security Audit Report:');
        this.vulnerabilities.forEach(vuln => {
            console.log(`- ${vuln.type} found in ${vuln.file}: ${vuln.occurrences} occurrence(s)`);
        });
    }

    // Run the security audit
    runAudit(directoryPath) {
        this.scanDirectory(directoryPath);
        this.generateReport();
    }
}

// Example usage
const audit = new SecurityAudit();
audit.runAudit('./src'); // Specify the directory to scan
