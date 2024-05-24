import { crypto } from 'crypto';

const encryptionKey = 'secret-key';

async function encryptData(data) {
  const cipher = crypto.createCipher('aes-256-cbc', encryptionKey);
  const encryptedData = cipher.update(data, 'utf8', 'hex');
  return encryptedData;
}

async function decryptData(encryptedData) {
  const decipher = crypto.createDecipher('aes-256-cbc', encryptionKey);
  const decryptedData = decipher.update(encryptedData, 'hex', 'utf8');
  return decryptedData;
}

export { encryptData, decryptData };
