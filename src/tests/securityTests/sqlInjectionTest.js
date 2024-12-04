// tests/securityTests/sqlInjectionTest.js

const axios = require('axios');

async function testSqlInjection() {
    const maliciousInput = "' OR '1'='1"; // Example of SQL injection payload
    try {
        const response = await axios.get(`http://localhost:3000/api/users?username=${maliciousInput}`);
        console.log('SQL Injection Test Response:', response.data);
    } catch (error) {
        console.error('SQL Injection Test Failed:', error.message);
    }
}

testSqlInjection();
