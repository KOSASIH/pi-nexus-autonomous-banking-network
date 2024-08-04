const CognitaToken = artifacts.require('CognitaToken');

contract('CognitaToken', accounts => {
    it('should have a total supply of 100 million', async () => {
        const token = await CognitaToken.deployed();
        const totalSupply = await token.totalSupply();
        assert.equal(totalSupply, 100000000);
    });

    it('should be able to transfer tokens', async () => {
        const token = await CognitaToken.deployed();
        const account1 = accounts[0];
        const account2 = accounts[1];
        await token.transfer(account2, 100);
        const balance = await token.balanceOf(account2);
        assert.equal(balance, 100);
    });
});
