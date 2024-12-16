// tests/securityTests/xssTest.js

const axios = require('axios');

async function testXSS() {
    const maliciousScript = '<script>alert("XSS")</script>'; // Example of XSS payload
    try {
        const response = await axios.get(`http://localhost:3000/api/users?input=${encodeURIComponent(maliciousScript)}`);
        console.log('XSS Test Response:', response.data);
    } catch (error) {
        console.error('XSS Test Failed:', error.message);
    }
}

testXSS();
