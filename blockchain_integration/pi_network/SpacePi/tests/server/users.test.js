const request = require('supertest');
const app = require('../server/app');
const bcrypt = require('bcrypt');

describe('Users API', () => {
  it('registers new user', async () => {
    const response = await request(app).post('/api/users/register').send({
      username: 'testuser',
      email: 'test@example.com',
      password: 'password123'
    });
    expect(response.status).toBe(201);
    expect(response.body.message).toBe('User created successfully');
  });

  it('logs in existing user', async () => {
    const response = await request(app).post('/api/users/login').send({
      email: 'test@example.com',
      password: 'password123'
    });
    expect(response.status).toBe(200);
    expect(response.body.token).toBeDefined();
  });
});
