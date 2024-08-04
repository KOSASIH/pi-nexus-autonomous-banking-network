import { SecurityUtils } from './security_utils';

class Security {
  constructor() {
    this.securityUtils = new SecurityUtils();
  }

  async initialize() {
    // Initialize security system
    await this.securityUtils.initialize();
  }

  async authenticate(credentials) {
    // Authenticate user with provided credentials
    const isAuthenticated = await this.securityUtils.authenticate(credentials);
    return isAuthenticated;
  }

  async authorize(user, permissions) {
    // Authorize user with provided permissions
    const isAuthorized = await this.securityUtils.authorize(user, permissions);
    return isAuthorized;
  }

  async generateToken(user) {
    // Generate a new authentication token for the user
    const token = await this.securityUtils.generateToken(user);
    return token;
  }

  async validateToken(token) {
    // Validate the provided authentication token
    const isValid = await this.securityUtils.validateToken(token);
    return isValid;
  }

  async logout(user) {
    // Log out the user and invalidate their authentication token
    await this.securityUtils.logout(user);
  }
}

export { Security };
