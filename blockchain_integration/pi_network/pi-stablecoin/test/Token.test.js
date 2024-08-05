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
        const initialBalance = await token.balanceOf('0x...AuthorizedAddress...');
        await token.mint('0x...AuthorizedAddress...', 100);
        const newBalance = await token.balanceOf('0x...AuthorizedAddress...');
        expect(newBalance).to.be.equal(initialBalance.add(100));
    });

    it('should transfer tokens', async () => {
        const initialBalanceFrom = await token.balanceOf('0x...AuthorizedAddress...');
        const initialBalanceTo = await token.balanceOf('0x...RecipientAddress...');
        await token.transfer('0x...AuthorizedAddress...', '0x...RecipientAddress...', 50);
        const newBalanceFrom = await token.balanceOf('0x...AuthorizedAddress...');
        const newBalanceTo = await token.balanceOf('0x...RecipientAddress...');
        expect(newBalanceFrom).to.be.equal(initialBalanceFrom.sub(50));
        expect(newBalanceTo).to.be.equal(initialBalanceTo.add(50));
    });

    it('should burn tokens', async () => {
        const initialBalance = await token.balanceOf('0x...AuthorizedAddress...');
        await token.burn(50);
        const newBalance = await token.balanceOf('0x...AuthorizedAddress...');
        expect(newBalance).to.be.equal(initialBalance.sub(50));
    });
});
