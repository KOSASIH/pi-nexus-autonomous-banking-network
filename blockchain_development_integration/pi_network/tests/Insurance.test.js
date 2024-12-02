const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Insurance Contract", function () {
    let Insurance, insurance, policyHolder;

    beforeEach(async function () {
        [policyHolder] = await ethers.getSigners();
        Insurance = await ethers.getContractFactory("Insurance");
        insurance = await Insurance.deploy();
        await insurance.deployed();
    });

    it("should allow a user to purchase a policy", async function () {
        await insurance.connect(policyHolder).purchasePolicy(ethers.utils.parseEther("1"), ethers.utils.parseEther("5"), { value: ethers.utils.parseEther("1") });
        const policy = await insurance.policies(1);
        expect(policy.policyHolder).to.equal(policyHolder.address);
        expect(policy.coverageAmount).to.equal(ethers.utils.parseEther("5"));
        expect(policy.isActive).to.be.true;
    });

    it("should allow a policy holder to claim a policy", async function () {
        await insurance.connect(policyHolder).purchasePolicy(ethers.utils.parseEther("1"), ethers.utils.parseEther("5"), { value: ethers.utils.parseEther("1") });
        await insurance.connect(policyHolder).claimPolicy(1);
        const policy = await insurance.policies(1);
        expect(policy.isActive).to.be.false;
    });
});
