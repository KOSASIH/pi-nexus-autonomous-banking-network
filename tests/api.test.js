// tests/api.test.js
const request = require('supertest');
const app = require('../app'); // Assuming you have an Express app

describe('API Module', () => {
    test('GET /api/users should return a list of users', async () => {
        const response = await request(app).get('/api/users');
        expect(response.status).toBe(200);
        expect(response.body).toBeInstanceOf(Array);
    });

    test('POST /api/users should create a new user', async () => {
        const response = await request(app)
            .post('/api/users')
            .send({ name: 'John Doe', email: 'john@example.com' });
        expect(response.status).toBe(201);
        expect(response.body.name).toBe('John Doe');
    });
});
