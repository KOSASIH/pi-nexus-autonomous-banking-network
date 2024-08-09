import * as bcrypt from 'bcrypt';
import * as crypto from 'crypto';
import * as elliptic from 'elliptic';
import * as scrypt from 'scrypt-js';
import { randomBytes } from 'crypto';
import { promisify } from 'util';

const ec = new elliptic.ec('secp256k1');

interface HashOptions {
  saltRounds?: number;
  pepper?: string;
}

interface EncryptOptions {
  algorithm?: string;
  password?: string;
  salt?: string;
  iv?: string;
}

interface DecryptOptions {
  algorithm?: string;
  password?: string;
  salt?: string;
  iv?: string;
}

class Crypto {
  private static async generateSalt(): Promise<string> {
    return promisify(crypto.randomBytes)(16).then((buf) => buf.toString('hex'));
  }

  private static async generatePepper(): Promise<string> {
    return promisify(crypto.randomBytes)(16).then((buf) => buf.toString('hex'));
  }

  static async hash(password: string, options: HashOptions = {}): Promise<string> {
    const saltRounds = options.saltRounds || 10;
    const pepper = options.pepper || await this.generatePepper();
    const salt = await this.generateSalt();
    const hash = await bcrypt.hash(password + pepper, saltRounds);
    return `${salt}:${hash}`;
  }

  static async verify(password: string, hash: string): Promise<boolean> {
    const [salt, storedHash] = hash.split(':');
    const pepper = await this.generatePepper();
    const hashedPassword = await bcrypt.hash(password + pepper, parseInt(salt, 10));
    return hashedPassword === storedHash;
  }

  static async encrypt(data: string, options: EncryptOptions = {}): Promise<string> {
    const algorithm = options.algorithm || 'aes-256-cbc';
    const password = options.password || randomBytes(32).toString('hex');
    const salt = options.salt || await this.generateSalt();
    const iv = options.iv || randomBytes(16).toString('hex');
    const cipher = crypto.createCipheriv(algorithm, password, iv);
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return `${salt}:${iv}:${encrypted}`;
  }

  static async decrypt(encrypted: string, options: DecryptOptions = {}): Promise<string> {
    const [salt, iv, encryptedData] = encrypted.split(':');
    const algorithm = options.algorithm || 'aes-256-cbc';
    const password = options.password || randomBytes(32).toString('hex');
    const decipher = crypto.createDecipheriv(algorithm, password, iv);
    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  static async sign(data: string, privateKey: string): Promise<string> {
    const key = ec.keyFromPrivate(privateKey, 'hex');
    const signature = key.sign(data);
    return signature.toDER('hex');
  }

  static async verifySignature(data: string, signature: string, publicKey: string): Promise<boolean> {
    const key = ec.keyFromPublic(publicKey, 'hex');
    return key.verify(data, signature);
  }

  static async scrypt(password: string, salt: string, N: number, r: number, p: number): Promise<string> {
    return scrypt.scrypt(password, salt, N, r, p);
  }
}

export default Crypto;
