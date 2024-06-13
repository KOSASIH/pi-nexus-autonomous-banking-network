import { CryptoJS } from 'crypto-js';

class CryptographicUtilities {
  constructor() {
    this.cryptoJs = new CryptoJS();
  }

  async encryptData(data, key) {
    const encryptedData = await this.cryptoJs.encrypt(data, key);
    return encryptedData;
  }

  async decryptData(encryptedData, key) {
    const decryptedData = await this.cryptoJs.decrypt(encryptedData, key);
    return decryptedData;
  }
}

export default CryptographicUtilities;
