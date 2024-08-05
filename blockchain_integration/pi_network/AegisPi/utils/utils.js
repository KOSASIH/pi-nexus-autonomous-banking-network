import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import * as jsonwebtoken from 'jsonwebtoken';
import * as bcrypt from 'bcrypt';
import * as moment from 'oment';

class Utils {
  constructor() {}

  async generateUUID() {
    // Generate a unique UUID
    return crypto.randomBytes(16).toString('hex');
  }

  async hashPassword(password) {
    // Hash a password using bcrypt
    const salt = await bcrypt.genSalt(10);
    const hash = await bcrypt.hash(password, salt);
    return hash;
  }

  async verifyPassword(password, hash) {
    // Verify a password against a hash
    return await bcrypt.compare(password, hash);
  }

  async generateToken(payload) {
    // Generate a JSON Web Token
    const token = jsonwebtoken.sign(payload, process.env.SECRET_KEY, {
      expiresIn: '1h'
    });
    return token;
  }

  async verifyToken(token) {
    // Verify a JSON Web Token
    try {
      const payload = jsonwebtoken.verify(token, process.env.SECRET_KEY);
      return payload;
    } catch (error) {
      return null;
    }
  }

  async compressData(data) {
    // Compress data using zlib
    return zlib.gzipSync(data);
  }

  async decompressData(data) {
    // Decompress data using zlib
    return zlib.gunzipSync(data);
  }

  async readFile(filePath) {
    // Read a file from disk
    return fs.readFileSync(filePath, 'utf8');
  }

  async writeFile(filePath, data) {
    // Write a file to disk
    fs.writeFileSync(filePath, data);
  }

  async getDirectoryFiles(directoryPath) {
    // Get a list of files in a directory
    return fs.readdirSync(directoryPath);
  }

  async getDirectorySize(directoryPath) {
    // Get the size of a directory
    let size = 0;
    const files = await this.getDirectoryFiles(directoryPath);
    for (const file of files) {
      const filePath = path.join(directoryPath, file);
      const stats = fs.statSync(filePath);
      size += stats.size;
    }
    return size;
  }

  async formatDateTime(date) {
    // Format a date and time using moment.js
    return moment(date).format('YYYY-MM-DD HH:mm:ss');
  }

  async sleep(ms) {
    // Sleep for a specified amount of time
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export { Utils };
