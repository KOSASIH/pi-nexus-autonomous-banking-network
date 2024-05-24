import { generateRandomToken } from "./utils";

class TwoFactorAuth {
  constructor(user) {
    this.user = user;
  }

  async sendVerificationCode() {
    const token = generateRandomToken();
    // Send token to user's phone or email
    await this.user.sendVerificationCode(token);
    return token;
  }

  async verifyCode(code) {
    if (code === this.user.verificationCode) {
      // Code is valid, allow sensitive transaction
      return true;
    } else {
      // Code is invalid, deny sensitive transaction
      return false;
    }
  }
}

export default TwoFactorAuth;
