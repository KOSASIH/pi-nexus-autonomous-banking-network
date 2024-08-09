/**
 * Unit tests for the authentication backend
 */

import { AuthBackend } from '../../src/backend/auth';
import { QCLOUD_API_KEY, QCLOUD_API_URL } from '../../src/qc-utils/qc-constants';

describe('AuthBackend', () => {
  const authBackend = new AuthBackend();

  beforeEach(() => {
    jest.resetModules();
  });

  it('should set the API key and URL', () => {
    authBackend.setApiKey(QCLOUD_API_KEY);
    authBackend.setApiUrl(QCLOUD_API_URL);

    expect(authBackend.apiKey).toBe(QCLOUD_API_KEY);
    expect(authBackend.apiUrl).toBe(QCLOUD_API_URL);
  });

  it('should authenticate successfully', async () => {
    const response = await authBackend.authenticate();

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('token');
  });

  it('should throw an error on authentication failure', async () => {
    jest.spyOn(global, 'fetch').mockImplementation(() => {
      return Promise.reject(new Error('Authentication failed'));
    });

    await expect(authBackend.authenticate()).rejects.toThrowError('Authentication failed');
  });

  it('should refresh the token successfully', async () => {
    const refreshToken = 'refresh-token';
    const newToken = 'new-token';

    jest.spyOn(global, 'fetch').mockImplementation(() => {
      return Promise.resolve({
        status: 200,
        data: { token: newToken },
      });
    });

    await authBackend.refreshToken(refreshToken);

    expect(authBackend.token).toBe(newToken);
  });

  it('should throw an error on token refresh failure', async () => {
    const refreshToken = 'refresh-token';

    jest.spyOn(global, 'fetch').mockImplementation(() => {
      return Promise.reject(new Error('Token refresh failed'));
    });

    await expect(authBackend.refreshToken(refreshToken)).rejects.toThrowError('Token refresh failed');
  });
});
