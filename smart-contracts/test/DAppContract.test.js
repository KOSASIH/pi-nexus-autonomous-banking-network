const DAppContract = artifacts.require("DAppContract");

contract("DAppContract", (accounts) => {
    let dAppContract;
    const [owner, user1] = accounts;

    beforeEach(async () => {
        dAppContract = await DAppContract.new("My DApp");
    });

    it("should set the correct name", async () => {
        const name = await dAppContract.name();
        assert.equal(name, "My DApp", "The name is not set correctly");
    });

    it("should allow users to deposit funds", async () => {
        const depositAmount = web3.utils.toWei("1", "ether");
        await dAppContract.deposit({ from: user1, value: depositAmount });

        const balance = await dAppContract.balances(user1);
        assert.equal(balance.toString(), depositAmount, "The balance is not updated correctly");
    });

    it("should allow users to withdraw funds", async () => {
        const depositAmount = web3.utils.toWei("1", "ether");
        await dAppContract.deposit({ from: user1, value: depositAmount });

        await dAppContract.withdraw(depositAmount, { from: user1 });
        const balance = await dAppContract.balances(user1);
        assert.equal(balance.toString(), "0", "The balance should be zero after withdrawal");
    });

    it("should not allow withdrawal of more than balance", async () => {
        try {
            await dAppContract.withdraw(web3.utils.toWei("1", "ether"), { from: user1 });
            assert.fail("Withdrawal should have failed");
        } catch (error) {
            assert(error.message.includes("Insufficient balance"), "Expected 'Insufficient balance' error");
        }
    });
});
