// File name: decentralized_identity_management.js
const ipfs = require("ipfs-api");

class DecentralizedIdentityManagement {
  async storeIdentity(identity) {
    const node = new ipfs({
      host: "ipfs.infura.io",
      port: 5001,
      protocol: "https",
    });
    const file = await node.add(identity);
    return file.hash;
  }

  async retrieveIdentity(hash) {
    const node = new ipfs({
      host: "ipfs.infura.io",
      port: 5001,
      protocol: "https",
    });
    const file = await node.cat(hash);
    return file.toString();
  }
}
