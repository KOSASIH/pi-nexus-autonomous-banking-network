const { expect } = require('chai');
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const User = artifacts.require('User');

contract('User', () => {
    let user;

    beforeEach(async () => {
        user = await User.new();
    });

    it('should add a user', async () => {
        await user.addUser('0x...NewUserAddress...');
        const isAdmin = await user.isAdmin('0x...NewUserAddress...');
        expect(isAdmin).to.be.true;
    });

    it('should remove a user', async () => {
        await user.addUser('0x...NewUserAddress...');
        await user.removeUser('0x...NewUserAddress...');
        const isAdmin = await user.isAdmin('0x...NewUserAddress...');
        expect(isAdmin).to.be.false;
    });
});
