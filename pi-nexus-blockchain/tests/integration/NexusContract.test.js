import { NexusContract } from '../../src/contracts/NexusContract';
import Web3 from 'web3';
import { expect } from 'chai';

describe('NexusContract', () => {
  let web3;
  let nexusContract;

  beforeEach(async () => {
    web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    nexusContract = new NexusContract(web3);
  });

  it('should deploy contract', async () => {
    const txCount = await web3.eth.getTransactionCount();
    const deployTx = await nexusContract.deploy();
    expect(deployTx.transactionHash).to.exist;
    expect(txCount + 1).to.equal(await web3.eth.getTransactionCount());
  });

  it('should mint tokens', async () => {
    const address = '0x1234567890abcdef';
    const amount = 100;
    const mintTx = await nexusContract.mint(address, amount);
    expect(mintTx.transactionHash).to.exist;
    const balance = await nexusContract.balanceOf(address);
    expect(balance).to.equal(amount);
  });

  it('should transfer tokens', async () => {
    const fromAddress = '0x1234567890abcdef';
    const toAddress = '0xabcdef1234567890';
    const amount = 50;
    const transferTx = await nexusContract.transfer(fromAddress, toAddress, amount);
    expect(transferTx.transactionHash).to.exist;
    const fromBalance = await nexusContract.balanceOf(fromAddress);
    expect(fromBalance).to.equal(50);
    const toBalance = await nexusContract.balanceOf(toAddress);
    expect(toBalance).to.equal(amount);
  });
});
