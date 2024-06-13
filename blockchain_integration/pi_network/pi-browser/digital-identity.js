import { uPort } from 'uport-credentials';

class DigitalIdentity {
  constructor() {
    this.uport = new uPort();
  }

  async createIdentity() {
    const identity = await this.uport.createIdentity();
    return identity;
  }

  async verifyIdentity() {
    const verification = await this.uport.verifyIdentity();
    return verification;
  }
}

export default DigitalIdentity;
