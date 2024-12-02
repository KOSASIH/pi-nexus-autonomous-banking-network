// LoanProcessing.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("LoanProcessing Contract", function () {
    let LoanProcessing;
    let loanProcessing;
    let owner;
    let addr1;
    let addr2;

    beforeEach(async function () {
        LoanProcessing = await ethers.getContractFactory("LoanProcessing");
        [owner, addr1, addr2] = await ethers.getSigners();
        loanProcessing = await LoanProcessing.deploy();
        await loanProcessing.deployed();
    });

    it("Should allow a user to apply for a loan", async function () {
        const amount = ethers.utils.parseEther("1.0");
        const interestRate = 5; // 5%
        const duration = 30; // 30 days

        await loanProcessing.connect(addr1).applyForLoan(amount, interestRate, duration);
        const loan = await loanProcessing.getLoanDetails(1);

        expect(loan.amount).to.equal(amount);
        expect(loan.interestRate).to.equal(interestRate);
        expect(loan.duration).to.equal(duration);
        expect(loan.status).to.equal(0); // 0 for pending
    });

    it("Should allow a user to repay a loan", async function () {
        const amount = ethers.utils.parseEther("1.0");
        const interestRate = 5;
        const duration = 30;

        await loanProcessing.connect(addr1).applyForLoan(amount, interestRate, duration);
        await loanProcessing.connect(addr1).repayLoan(1, { value: amount });

        const loan = await loanProcessing.getLoanDetails(1);
        expect(loan.status).to.equal(1); // 1 for repaid
    });

    it("Should revert if a user tries to repay a non-existent loan", async function () {
        await expect(loanProcessing.connect(addr1).repayLoan(999)).to.be.revertedWith("Loan does not exist");
    });

    it("Should allow the owner to withdraw funds", async function () {
        const amount = ethers.utils.parseEther("1.0");
        await loanProcessing.connect(addr1).applyForLoan(amount, 5, 30);
        await loanProcessing.connect(addr1).repayLoan(1, { value: amount });

        const initialBalance = await ethers.provider.getBalance(owner.address);
        await loanProcessing.connect(owner).withdrawFunds();
        const finalBalance = await ethers.provider.getBalance(owner.address);

        expect(finalBalance).to.be.gt(initialBalance); // Owner's balance should increase
    });
});
