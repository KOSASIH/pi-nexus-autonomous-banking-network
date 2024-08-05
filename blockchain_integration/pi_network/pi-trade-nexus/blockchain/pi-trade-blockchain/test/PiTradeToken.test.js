const Web3 = require('web3');
const { PiTradeToken } = require('../build/contracts');
const { expect } = require('chai');

describe('PiTradeToken', () => {
  let web3;
  let piTradeToken;
  let accounts;

  beforeEach(async () => {
    web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));
    accounts = await web3.eth.getAccounts();
    piTradeToken = await PiTradeToken.new(web3.utils.toWei('100000000', 'ether'), {
      from: accounts[0],
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });
  });

  it('should have a total supply of 100 million tokens', async () => {
    const totalSupply = await piTradeToken.methods.totalSupply().call();
    expect(totalSupply).to.equal(web3.utils.toWei('100000000', 'ether'));
  });

  it('should allow token transfer', async () => {
    const sender = accounts[0];
    const recipient = accounts[1];
    const amount = web3.utils.toWei('100', 'ether');

    await piTradeToken.methods.transfer(recipient, amount).send({
      from: sender,
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const senderBalance = await piTradeToken.methods.balanceOf(sender).call();
    const recipientBalance = await piTradeToken.methods.balanceOf(recipient).call();

    expect(senderBalance).to.equal(web3.utils.toWei('99999900', 'ether'));
    expect(recipientBalance).to.equal(amount);
  });

  it('should allow token approval', async () => {
    const owner = accounts[0];
    const spender = accounts[1];
    const amount = web3.utils.toWei('100', 'ether');

    await piTradeToken.methods.approve(spender, amount).send({
      from: owner,
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const allowance = await piTradeToken.methods.allowance(owner, spender).call();
    expect(allowance).to.equal(amount);
  });
});
