const { FabricCAServices } = require('fabric-ca-client');
const { Wallets } = require('fabric-network');
const { Gateway } = require('fabric-network');

const caUrl = 'https://localhost:7054';
const walletPath = './wallet';

const ca = new FabricCAServices(caUrl);
const wallet = await Wallets.newFileSystemWallet(walletPath);

async function getAccountBalance(accountId) {
  const gateway = new Gateway();
  try {
    await gateway.connect(ccp, {
      wallet,
      identity: 'admin',
      discovery: { enabled: true, asLocalhost: true },
    });

    const network = await gateway.getNetwork('mychannel');
    const contract = network.getContract('pi-network');

    const result = await contract.evaluateTransaction('getAccountBalance', accountId);
    return result.toString();
  } finally {
    gateway.disconnect();
  }
}

async function getTransactionHistory(accountId) {
  const gateway = new Gateway();
  try {
    await gateway.connect(ccp, {
      wallet,
      identity: 'admin',
      discovery: { enabled: true, asLocalhost: true },
    });

    const network = await gateway.getNetwork('mychannel');
    const contract = network.getContract('pi-network');

    const result = await contract.evaluateTransaction('getTransactionHistory', accountId);
    return JSON.parse(result.toString());
  } finally {
    gateway.disconnect();
  }
}

async function createAccount(name, email) {
  const gateway = new Gateway();
  try {
    await gateway.connect(ccp, {
      wallet,
      identity: 'admin',
      discovery: { enabled: true, asLocalhost: true },
    });

    const network = await gateway.getNetwork('mychannel');
    const contract = network.getContract('pi-network');

    constresult = await contract.submitTransaction('createAccount', name, email);
    return JSON.parse(result.toString());
  } finally {
    gateway.disconnect();
  }
}

async function transferFunds(fromAccountId, toAccountId, amount) {
  const gateway = new Gateway();
  try {
    await gateway.connect(ccp, {
      wallet,
      identity: 'admin',
      discovery: { enabled: true, asLocalhost: true },
    });

    const network = await gateway.getNetwork('mychannel');
    const contract = network.getContract('pi-network');

    const result = await contract.submitTransaction('transferFunds', fromAccountId, toAccountId, amount);
    return JSON.parse(result.toString());
  } finally {
    gateway.disconnect();
  }
}

module.exports = {
  getAccountBalance,
  getTransactionHistory,
  createAccount,
  transferFunds,
};
