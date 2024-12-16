// tests/test_deploy.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Deployment Tests", function () {
    let MultiSigWallet, GovernanceContract;
    let multiSigWallet, governanceContract;
    let owners;

    before(async function () {
        owners = [ethers.Wallet.createRandom().address, ethers.Wallet.createRandom().address];
        const requiredConfirmations = 2;

        MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
        multiSigWallet = await MultiSigWallet.deploy(owners, requiredConfirmations);
        await multiSigWallet.deployed();

        GovernanceContract = await ethers.getContractFactory("GovernanceContract");
        governanceContract = await GovernanceContract.deploy(multiSigWallet.address);
        await governanceContract.deployed();
    });

    it("should deploy MultiSigWallet", async function () {
        expect(multiSigWallet.address).to.properAddress;
    });

    it("should deploy GovernanceContract", async function () {
        expect(governanceContract.address).to.properAddress;
    });
});
