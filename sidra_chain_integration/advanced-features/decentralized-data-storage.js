const IPFS = require('ipfs');

async function storeData(userInput) {
  const ipfs = new IPFS();
  const result = await ipfs.add(userInput);
  return result;
}

module.exports = { storeData };
