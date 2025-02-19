// tests/userService.test.ts
import request from 'supertest';
import app from '../src/app'; // Assuming you have an Express app

describe('User  Service', () => {
    it('should register a user', async () => {
        const response = await request(app).post('/api/users/register').send({
            username: 'testuser',
            password: 'testpassword',
        });
        expect(response.status).toBe(201);
        expect(response.body.message).toBe('User  registered successfully');
    });
});
