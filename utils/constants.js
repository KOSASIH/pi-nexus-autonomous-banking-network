// utils/constants.js

// API Endpoints
export const API_BASE_URL = 'https://api.decentralizedbanking.com';
export const REGISTER_ENDPOINT = '/register';
export const LOGIN_ENDPOINT = '/login';
export const CREATE_IDENTITY_ENDPOINT = '/create-identity';
export const DEPOSIT_ENDPOINT = '/deposit';
export const WITHDRAW_ENDPOINT = '/withdraw';
export const CREATE_ESCROW_ENDPOINT = '/create-escrow';
export const RESOLVE_ESCROW_ENDPOINT = '/resolve-escrow';

// Web3
export const WEB3_PROVIDER_URL = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID';

// Contract Addresses
export const BANKING_CONTRACT_ADDRESS = 'CONTRACT_ADDRESS';

// JWT
export const JWT_SECRET_KEY = 'SECRET_KEY';

// Rate Limiting
export const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes
export const RATE_LIMIT_MAX = 100; // Limit each IP to 100 requests per windowMs

// Risk Management
export const RISK_MANAGEMENT_THRESHOLD = 0.8; // Threshold for triggering risk management

// Compliance and Regulatory Framework
export const AML_REGEX = /^[A-Z0-9]+$/; // Regular expression for validating AML-compliant identifiers
export const KYC_REGEX = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/; // Regular expression for validating KYC-compliant email addresses
