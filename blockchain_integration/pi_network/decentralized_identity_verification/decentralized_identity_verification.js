// decentralized_identity_verification.js
const { IdentityManager } = require("erc-725");

const identityManager = new IdentityManager();

async function verifyIdentity(userUid, metadata) {
  const identity = await identityManager.getIdentity(userUid);
  if (identity) {
    // Verify identity using metadata
    const isValid = await identityManager.verifyIdentity(identity, metadata);
    return isValid;
  } else {
    return false;
  }
}
