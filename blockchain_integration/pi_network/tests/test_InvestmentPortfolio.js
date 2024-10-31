// tests/test_InvestmentPortfolio.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("InvestmentPortfolio", function () {
    let InvestmentPortfolio;
    let investmentPortfolio;
    let owner;
    let addr1;
    let token;

    beforeEach(async function () {
        // Deploy a mock ERC20 token for testing
        const Token = await ethers.getContractFactory("MockERC20");
        token = await Token.deploy("Test Token", "TTK", ethers.utils.parseEther("1000"));
        await token.deployed();

        InvestmentPortfolio = await ethers.getContractFactory("InvestmentPortfolio");
        [owner, addr1] = await ethers.getSigners();
        investmentPortfolio = await InvestmentPortfolio.deploy();
        await investmentPortfolio.deployed();
    });

    it("Should allow investment in the portfolio", async function () {
        const investmentAmount = ethers.utils.parseEther("100.0");

        // Approve the investment amount
        await token.approve(investmentPortfolio.address, investmentAmount);
        
        // Invest in the portfolio
        await investmentPortfolio.invest(token.address, investmentAmount);
        
        // Check the investment amount
        expect(await investmentPortfolio.getInvestmentAmount(token.address)).to.equal(investmentAmount);
    });

    it("Should allow withdrawal of investment", async function () {
        const investmentAmount = ethers.utils.parseEther("100.0");

        // Approve and invest in the portfolio
        await token.approve(investmentPortfolio.address, investmentAmount);
        await investmentPortfolio.invest(token.address, investmentAmount);
        
        // Withdraw the investment
        await investmentPortfolio.withdrawInvestment(token.address, investmentAmount);
        
        // Check the investment amount
        expect(await investmentPortfolio.getInvestmentAmount(token.address)).to.equal(0);
    });

    it("Should not allow withdrawal of more than invested amount", async function () {
        const investmentAmount = ethers.utils.parseEther("100.0");

        // Approve and invest in the portfolio
        await token.approve(investmentPortfolio.address, investmentAmount);
        await investmentPortfolio.invest(token.address, investmentAmount);
        
        // Attempt to withdraw more than invested
        await expect(investmentPortfolio.withdrawInvestment(token.address, ethers.utils.parseEther("200.0"))).to.be.revertedWith("Insufficient investment amount");
    });

    it("Should track multiple investments", async function () {
        const investmentAmount1 = ethers.utils.parseEther("100.0");
        const investmentAmount2 = ethers.utils.parseEther("200.0");

        // Approve and invest in the portfolio
        await token.approve(investmentPortfolio.address, investmentAmount1);
        await investmentPortfolio.invest(token.address, investmentAmount1);
        
        // Approve and invest in another asset
        const token2 = await token.deploy("Another Token", "ATK", ethers.utils.parseEther("1000"));
        await token2.deployed();
        await token2.approve(investmentPortfolio.address, investmentAmount2);
        await investmentPortfolio.invest(token2.address, investmentAmount2);
        
        // Check the investment amounts
        expect(await investmentPortfolio.getInvestmentAmount(token.address)).to.equal(investmentAmount1);
        expect(await investmentPortfolio.getInvestmentAmount(token2.address)).to.equal(investmentAmount2);
    });
});
