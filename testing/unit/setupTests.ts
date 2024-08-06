// setupTests.ts
import { configure } from 'jest';

configure({
  verbose: true,
});

jest.mock('axios', () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
}));
