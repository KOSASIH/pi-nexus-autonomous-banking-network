import { LOCAL_STORAGE_TOKEN_KEY, LOCAL_STORAGE_USER_KEY } from './constants';

class Helpers {
  static getLocalStorageItem(key) {
    return JSON.parse(localStorage.getItem(key));
  }

  static setLocalStorageItem(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
  }

  static removeLocalStorageItem(key) {
    localStorage.removeItem(key);
  }

  static formatCurrency(value, currencySymbol) {
    return `${value.toFixed(2)} ${currencySymbol}`;
  }

  static formatDateTime(date) {
    return date.toLocaleString();
  }

  static generateRandomString(length) {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
  }

  static isTokenValid(token) {
    try {
      const decodedToken = jwt.verify(token, process.env.SECRET_KEY);
      return decodedToken.exp > Date.now() / 1000;
    } catch (error) {
      return false;
    }
  }

  static getUserFromLocalStorage() {
    return this.getLocalStorageItem(LOCAL_STORAGE_USER_KEY);
  }

  static getTokenFromLocalStorage() {
    return this.getLocalStorageItem(LOCAL_STORAGE_TOKEN_KEY);
  }
}

export default Helpers;
