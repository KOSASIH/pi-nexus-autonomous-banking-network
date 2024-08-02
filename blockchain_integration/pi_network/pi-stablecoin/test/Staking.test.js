const { expect } = require('chai');
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const Staking = artifacts.require('Staking');

contract('Staking', () => {
    let staking;

    beforeEach(async () => {
        staking = await Staking.new();
    });

    it('should add a staker', async () => {
        await staking.addStaker('0x...NewStakerAddress...');
        const isStaker = await staking.isStaker('0x...NewStakerAddress...');
        expect(isStaker).to.be.true;
    });

        it('should update a staker's balance', async () => {
        await staking.addStaker('0x...NewStakerAddress...');
        const initialBalance = await staking.getStakerBalance('0x...NewStakerAddress...');
        await staking.updateStakerBalance('0x...NewStakerAddress...', 100);
        const newBalance = await staking.getStakerBalance('0x...NewStakerAddress...');
        expect(newBalance).to.be.equal(initialBalance.add(100));
    });
});
   
