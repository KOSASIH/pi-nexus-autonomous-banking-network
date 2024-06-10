const uPort = require('uport');

class PiNexusIdentityManager {
    constructor() {
        this.uport = new uPort('https://api.uport.me');
    }

    async createUserIdentity(did, attributes) {
        const identity = await this.uport.createIdentity(did, attributes);
        return identity;
    }

    async updateUserIdentity(did, attributes) {
        const identity = await this.uport.updateIdentity(did, attributes);
        return identity;
    }

    async getUserIdentity(did) {
        const identity = await this.uport.getIdentity(did);
        return identity;
    }
}

module.exports = PiNexusIdentityManager;
