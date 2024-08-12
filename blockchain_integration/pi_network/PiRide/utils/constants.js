export const API_URL = 'https://api.example.com';
export const API_VERSION = 'v1';

export const ROLE_ADMIN = 'admin';
export const ROLE_USER = 'user';

export const STATUS_PENDING = 'pending';
export const STATUS_ACCEPTED = 'accepted';
export const STATUS_REJECTED = 'rejected';
export const STATUS_COMPLETED = 'completed';

export const RIDE_TYPE_ONE_WAY = 'one-way';
export const RIDE_TYPE_ROUND_TRIP = 'round-trip';

export const MAX_RIDE_REQUESTS = 5;
export const MAX_RIDE_OFFERS = 10;

export const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
export const PASSWORD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

export const JWT_SECRET_KEY = 'secret-key';
export const JWT_EXPIRATION_TIME = '1h';

export const MONGODB_URI = 'mongodb://localhost:27017/ride-sharing';
export const MONGODB_DB_NAME = 'ride-sharing';
