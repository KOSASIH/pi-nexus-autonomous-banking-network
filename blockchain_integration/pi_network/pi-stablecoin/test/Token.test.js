const { expect } = require('chai');
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const Token = artifacts.require('Token');

contract('Token', () => {
    let token;

    beforeEach(async () => {
        token = await Token.new();
    });

    it('should mint tokens', async () => {
        const
