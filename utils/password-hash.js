// password-hash.js
const bcrypt = require('bcrypt');

class PasswordHash {
  async hash(password) {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(password, salt);
  }

  async compare(password, hash) {
    return await bcrypt.compare(password, hash);
  }
}

module.exports = PasswordHash;
