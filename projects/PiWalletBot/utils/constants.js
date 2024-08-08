export const API_URL = 'https://api.example.com';
export const LOCAL_STORAGE_TOKEN_KEY = 'token';
export const LOCAL_STORAGE_USER_KEY = 'user';

export const TRANSACTION_STATUS = {
  PENDING: 'pending',
  SUCCESSFUL: 'successful',
  FAILED: 'failed',
};

export const WALLET_CURRENCIES = [
  { symbol: 'PIC', name: 'Pi Coin' },
  { symbol: 'BTC', name: 'Bitcoin' },
  { symbol: 'ETH', name: 'Ethereum' },
];

export const ERROR_MESSAGES = {
  INVALID_CREDENTIALS: 'Invalid username or password',
  INSUFFICIENT_BALANCE: 'Insufficient balance',
  TRANSACTION_FAILED: 'Transaction failed',
};

export const SUCCESS_MESSAGES = {
  TRANSACTION_SUCCESSFUL: 'Transaction successful',
  LOGIN_SUCCESSFUL: 'Login successful',
  REGISTER_SUCCESSFUL: 'Registration successful',
};
