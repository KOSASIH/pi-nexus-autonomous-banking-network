import * as CryptoJS from 'crypto-js';

class AstralPlaneSecurity {
  constructor() {
    this.key = 'secretkey';
  }

  async encrypt(data) {
    const encryptedData = CryptoJS.AES.encrypt(data, this.key);
    return encryptedData.toString();
  }

  async decrypt(encryptedData) {
    const decryptedData = CryptoJS.AES.decrypt(encryptedData, this.key);
    return decryptedData.toString(CryptoJS.enc.Utf8);
  }

  async hashPassword(password) {
    const hashedPassword = CryptoJS.SHA256(password);
    return hashedPassword.toString();
  }

  async verifyPassword(password, hashedPassword) {
    const hashedInput = CryptoJS.SHA256(password);
    return hashedInput.toString() === hashedPassword;
  }
}

export default AstralPlaneSecurity;
