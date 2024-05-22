const piPromise = require('pi-promise');
const networkInterfaces = piPromise.networkInterfaces;

class PiNetworkAdapter {
  constructor() {
    this.interfaces = {};
  }

  async initialize() {
    const ifaces = await networkInterfaces();
    for (const iface of ifaces) {
      this.interfaces[iface.name] = iface;
    }
  }

  getInterfaces() {
    return this.interfaces;
  }

  getInterfaceByName(name) {
    return this.interfaces[name];
  }

  async setInterfaceUp(name) {
    const iface = this.getInterfaceByName(name);
    if (iface) {
      iface.enabled = true;
      await iface.save();
    } else {
      throw new Error(`Network interface "${name}" not found.`);
    }
  }

  async setInterfaceDown(name) {
    const iface = this.getInterfaceByName(name);
    if (iface) {
      iface.enabled = false;
      await iface.save();
    } else {
      throw new Error(`Network interface "${name}" not found.`);
    }
  }
}

module.exports = PiNetworkAdapter;
