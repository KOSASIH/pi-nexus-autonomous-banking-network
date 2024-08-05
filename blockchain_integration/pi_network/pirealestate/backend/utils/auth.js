const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const { v4: uuidv4 } = require("uuid");
const crypto = require("crypto");
const { promisify } = require("util");

const signToken = promisify(jwt.sign);
const verifyToken = promisify(jwt.verify);
const hashPassword = promisify(bcrypt.hash);
const comparePassword = promisify(bcrypt.compare);

const AUTH_SECRET = process.env.AUTH_SECRET;
const REFRESH_SECRET = process.env.REFRESH_SECRET;
const TOKEN_EXPIRATION = process.env.TOKEN_EXPIRATION;
const REFRESH_TOKEN_EXPIRATION = process.env.REFRESH_TOKEN_EXPIRATION;

class AuthUtil {
  async generateToken(user) {
    const token = await signToken({ _id: user._id, role: user.role }, AUTH_SECRET, {
      expiresIn: TOKEN_EXPIRATION,
    });
    return token;
  }

  async generateRefreshToken(user) {
    const refreshToken = await signToken({ _id: user._id, role: user.role }, REFRESH_SECRET, {
      expiresIn: REFRESH_TOKEN_EXPIRATION,
    });
    return refreshToken;
  }

  async verifyToken(token) {
    try {
      const decoded = await verifyToken(token, AUTH_SECRET);
      return decoded;
    } catch (error) {
      throw error;
    }
  }

  async verifyRefreshToken(refreshToken) {
    try {
      const decoded = await verifyToken(refreshToken, REFRESH_SECRET);
      return decoded;
    } catch (error) {
      throw error;
    }
  }

  async hashPassword(password) {
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await hashPassword(password, salt);
    return hashedPassword;
  }

  async comparePassword(candidatePassword, hashedPassword) {
    const isMatch = await comparePassword(candidatePassword, hashedPassword);
    return isMatch;
  }

  async generateUUID() {
    return uuidv4();
  }

  async generateRandomBytes(size) {
    return crypto.randomBytes(size);
  }
}

module.exports = new AuthUtil();
