// tests/backend/app.test.js
import { jest } from '@jest/globals';
import app from '../app';
import request from 'supertest';
import { Web3Provider } from '../contexts/Web3Context';
import { PaymentGatewayProvider } from '../contexts/PaymentGatewayContext';
import { ethers } from 'ethers';

jest.setTimeout(30000); // 30 seconds

describe('App', () => {
  let web3Provider;
  let paymentGatewayProvider;

  beforeEach(async () => {
    web3Provider = new Web3Provider();
    paymentGatewayProvider = new PaymentGatewayProvider();
    await web3Provider.init();
    await paymentGatewayProvider.init();
  });

  afterEach(async () => {
    await web3Provider.close();
    await paymentGatewayProvider.close();
  });

  it('should return 200 OK on GET /', async () => {
    const response = await request(app).get('/');
    expect(response.status).toBe(200);
  });

  it('should return 404 Not Found on GET /unknown', async () => {
    const response = await request(app).get('/unknown');
    expect(response.status).toBe(404);
  });

  it('should process payment successfully', async () => {
    const amount = ethers.utils.parseEther('1.0');
    const payer = '0x1234567890abcdef';
    const payee = '0xabcdef1234567890';
    const paymentMethod = 'contract';

    const response = await request(app)
      .post('/payment')
      .send({ amount, payer, payee, paymentMethod });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: 'Payment successful!' });
  });

  it('should return error on invalid payment request', async () => {
    const amount = ethers.utils.parseEther('1.0');
    const payer = '0x1234567890abcdef';
    const payee = '0xabcdef1234567890';
    const paymentMethod = 'invalid';

    const response = await request(app)
      .post('/payment')
      .send({ amount, payer, payee, paymentMethod });

    expect(response.status).toBe(400);
    expect(response.body).toEqual({ error: 'Invalid payment method' });
  });
});
