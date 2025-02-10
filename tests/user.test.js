const request = require('supertest');
const app = require('../src/api/index'); // Adjust the path as necessary
const UserModel = require('../src/api/models/userModel');

describe('User API', () => {
    beforeAll(async () => {
        await UserModel.deleteMany(); // Clear the database before tests
    });

    it('should create a new user', async () => {
        const res = await request(app)
            .post('/api/users')
            .send({
                username: 'testuser',
                email: 'test@example.com',
                password: 'password123',
            });
        expect(res.statusCode).toEqual(201);
        expect(res.body.success).toBe(true);
    });

    it('should get a user by ID', async () => {
        const user = await UserModel.findOne({ email: 'test@example.com' });
        const res = await request(app).get(`/api/users/${user._id}`);
        expect(res.statusCode).toEqual(200);
        expect(res.body.success).toBe(true);
        expect(res.body.user.username).toBe('testuser');
    });
});
