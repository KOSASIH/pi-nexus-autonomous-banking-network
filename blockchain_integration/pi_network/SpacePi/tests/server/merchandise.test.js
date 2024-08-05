const request = require('supertest');
const app = require('../server/app');

describe('Merchandise API', () => {
  it('returns merchandise list', async () => {
    const response = await request(app).get('/api/space-x/merchandise');
    expect(response.status).toBe(200);
    expect(response.body.length).toBeGreaterThan(0);
  });

  it('returns merchandise by ID', async () => {
    const response = await request(app).get('/api/space-x/merchandise/1');
    expect(response.status).toBe(200);
    expect(response.body.id).toBe(1);
  });
});
