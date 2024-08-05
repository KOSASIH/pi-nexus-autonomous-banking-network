const request = require('supertest');
const app = require('../server/app');

describe('Launches API', () => {
  it('returns launch list', async () => {
    const response = await request(app).get('/api/space-x/launches');
    expect(response.status).toBe(200);
    expect(response.body.length).toBeGreaterThan(0);
  });

  it('returns launch by ID', async () => {
    const response = await request(app).get('/api/space-x/launches/1');
    expect(response.status).toBe(200);
    expect(response.body.id).toBe(1);
  });
});
