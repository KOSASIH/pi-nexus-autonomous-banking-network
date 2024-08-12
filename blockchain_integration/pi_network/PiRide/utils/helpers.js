import { API_URL } from './constants';

export function generateUUID() {
  return crypto.randomBytes(16).toString('hex');
}

export function validateEmail(email) {
  return EMAIL_REGEX.test(email);
}

export function validatePassword(password) {
  return PASSWORD_REGEX.test(password);
}

export function generateJWTToken(userId, role) {
  const token = jwt.sign({ userId, role }, JWT_SECRET_KEY, {
    expiresIn: JWT_EXPIRATION_TIME,
  });
  return token;
}

export function verifyJWTToken(token) {
  try {
    const decoded = jwt.verify(token, JWT_SECRET_KEY);
    return decoded;
  } catch (error) {
    return null;
  }
}

export function makeAPIRequest(method, endpoint, data) {
  const url = `${API_URL}/${API_VERSION}/${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
  };

  if (method === 'GET') {
    return axios.get(url, { headers });
  } else if (method === 'POST') {
    return axios.post(url, data, { headers });
  } else if (method === 'PATCH') {
    return axios.patch(url, data, { headers });
  } else if (method === 'DELETE') {
    return axios.delete(url, { headers });
  }
}

export function formatRideData(ride) {
  return {
    id: ride._id,
    userId: ride.userId,
    pickupLocation: ride.pickupLocation,
    dropoffLocation: ride.dropoffLocation,
    rideDate: ride.rideDate,
    rideTime: ride.rideTime,
    rideType: ride.rideType,
    seatsAvailable: ride.seatsAvailable,
    price: ride.price,
    status: ride.status,
  };
}

export function formatUserData(user) {
  return {
    id: user._id,
    name: user.name,
    email: user.email,
    phoneNumber: user.phoneNumber,
    profilePicture: user.profilePicture,
    rideHistory: user.rideHistory,
  };
}
