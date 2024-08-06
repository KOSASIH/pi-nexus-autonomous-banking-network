const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');

const generateToken = (user) => {
  const token = jwt.sign({
    id: user.id,
    email: user.email,
    role: user.role,
  }, process.env.SECRET_KEY, {
    expiresIn: '1h',
  });
  return token;
};

const verifyToken = (req, res, next) => {
  const token = req.header('Authorization');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }
  try {
    const decoded = jwt.verify(token, process.env.SECRET_KEY);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(400).json({ message: 'Invalid token.' });
  }
};

const hashPassword = (password) => {
  const salt = bcrypt.genSaltSync(10);
  const hash = bcrypt.hashSync(password, salt);
  return hash;
};

const comparePassword = (password, hash) => {
  const isValid = bcrypt.compareSync(password, hash);
  return isValid;
};

const generateOTP = () => {
  const otp = crypto.randomBytes(4).toString('hex');
  return otp;
};

module.exports = {
  generateToken,
  verifyToken,
  hashPassword,
  comparePassword,
  generateOTP,
};
