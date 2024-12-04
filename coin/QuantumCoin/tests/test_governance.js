// tests/test_governance.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Governance Tests", function () {
    ```javascript
    let GovernanceContract, governanceContract;
    let MultiSigWallet, multiSigWallet;
    let owner1, owner2, owner3;

    before(async function () {
        [owner1, owner2, owner3] = await ethers.getSigners();
        const owners = [owner1.address, owner2.address, owner3.address];
        const requiredConfirmations = 2;

        MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
        multiSigWallet = await MultiSigWallet.deploy(owners, requiredConfirmations);
        await multiSigWallet.deployed();

        GovernanceContract = await ethers.getContractFactory("GovernanceContract");
        governanceContract = await GovernanceContract.deploy(multiSigWallet.address);
        await governanceContract.deployed();
    });

    it("should create a proposal", async function () {
        const proposalTx = await governanceContract.createProposal("Increase funding", ethers.utils.parseEther("10"));
        await proposalTx.wait();

        const proposal = await governanceContract.proposals(0);
        expect(proposal.description).to.equal("Increase funding");
    });

    it("should allow voting on a proposal", async function () {
        await governanceContract.connect(owner1).vote(0, true);
        const proposal = await governanceContract.proposals(0);
        expect(proposal.votesFor).to.equal(1);
    });

    it("should execute a proposal after voting", async function () {
        await governanceContract.connect(owner2).vote(0, true);
        await governanceContract.executeProposal(0);
        const proposal = await governanceContract.proposals(0);
        expect(proposal.executed).to.be.true;
    });
});
