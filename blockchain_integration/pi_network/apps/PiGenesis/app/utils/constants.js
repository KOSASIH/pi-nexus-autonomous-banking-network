export const APP_NAME = 'PiGenesis';
export const APP_VERSION = '1.0.0';
export const API_URL = 'https://api.pigenesis.io';
export const BLOCKCHAIN_NETWORK = 'pi-network';
export const CURRENCY_SYMBOL = 'π';
export const DEFAULT_LANGUAGE = 'en-US';
export const SUPPORTED_LANGUAGES = ['en-US', 'es-ES', 'fr-FR', 'zh-CN'];
export const TIMEZONE = 'UTC';

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  USER_DATA: 'user_data',
  SETTINGS: 'settings',
};

export const ERROR_CODES = {
  INVALID_CREDENTIALS: 401,
  INSUFFICIENT_BALANCE: 402,
  TRANSACTION_FAILED: 403,
  NETWORK_ERROR: 500,
};

export const TRANSACTION_TYPES = {
  DEPOSIT: 'deposit',
  WITHDRAWAL: 'withdrawal',
  TRANSFER: 'transfer',
};
