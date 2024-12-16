// PenetrationTesting.js

const axios = require('axios');
const { URL } = require('url');

class PenetrationTesting {
    constructor(targetUrl) {
        this.targetUrl = targetUrl;
    }

    // Function to perform a basic HTTP GET request
    async httpGet(endpoint) {
        try {
            const response = await axios.get(new URL(endpoint, this.targetUrl).href);
            console.log(`GET ${endpoint} - Status: ${response.status}`);
            return response.data;
        } catch (error) {
            console.error(`GET ${endpoint} - Error: ${error.message}`);
        }
    }

    // Function to check for common vulnerabilities
    async checkCommonVulnerabilities() {
        console.log('Checking for common vulnerabilities...');

        // Check for open redirects
        await this.httpGet('/redirect?url=http://malicious-site.com');

        // Check for SQL injection
        await this.httpGet('/search?q=1%27%20OR%201=1--');

        // Check for Cross-Site Scripting (XSS)
        await this.httpGet('/comment?text=<script>alert("XSS")</script>');
    }

    // Function to run the penetration test
    async run() {
        console.log(`Starting penetration test on ${this.targetUrl}`);
        await this.checkCommonVulnerabilities();
        console.log('Penetration test completed.');
    }
}

// Example usage
const targetUrl = 'http://example.com'; // Replace with the target URL
const penTest = new PenetrationTesting(targetUrl);
penTest.run();
