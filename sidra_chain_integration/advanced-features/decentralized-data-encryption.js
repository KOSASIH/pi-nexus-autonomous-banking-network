const encryption = require('encryption-library');

async function encryptData(dataInput) {
  const encryptedData = await encryption.encrypt(dataInput);
  return encryptedData;
}

module.exports = { encryptData };
