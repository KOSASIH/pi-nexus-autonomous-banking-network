// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract InvestmentPortfolio is Ownable {
    struct Investment {
        address asset; // Address of the asset (ERC20 token)
        uint256 amount; // Amount invested
        uint256 purchasePrice; // Price at which the asset was purchased
        uint256 purchaseTime; // Timestamp of the purchase
    }

    struct Portfolio {
        address owner;
        Investment[] investments;
        uint256 totalInvested;
    }

    mapping(address => Portfolio) public portfolios;

    event InvestmentMade(address indexed user, address indexed asset, uint256 amount, uint256 price);
    event InvestmentWithdrawn(address indexed user, address indexed asset, uint256 amount);
    event PortfolioClosed(address indexed user);

    function invest(address asset, uint256 amount) public {
        require(amount > 0, "Investment amount must be greater than zero.");
        require(IERC20(asset).balanceOf(msg.sender) >= amount, "Insufficient balance.");

        // Transfer tokens from the user to the contract
        IERC20(asset).transferFrom(msg.sender, address(this), amount);

        // Record the investment
        uint256 price = getCurrentPrice(asset); // Assume this function gets the current price of the asset
        portfolios[msg.sender].investments.push(Investment({
            asset: asset,
            amount: amount,
            purchasePrice: price,
            purchaseTime: block.timestamp
        }));

        portfolios[msg.sender].totalInvested += amount;

        emit InvestmentMade(msg.sender, asset, amount, price);
    }

    function withdrawInvestment(address asset, uint256 amount) public {
        Portfolio storage portfolio = portfolios[msg.sender];
        require(portfolio.owner == msg.sender, "Only the portfolio owner can withdraw.");
        require(amount > 0, "Withdrawal amount must be greater than zero.");

        uint256 totalWithdrawable = getWithdrawableAmount(asset);
        require(amount <= totalWithdrawable, "Insufficient withdrawable amount.");

        // Find the investment and reduce the amount
        for (uint256 i = 0; i < portfolio.investments.length; i++) {
            if (portfolio.investments[i].asset == asset) {
                require(portfolio.investments[i].amount >= amount, "Not enough invested in this asset.");
                portfolio.investments[i].amount -= amount;

                // Transfer the tokens back to the user
                IERC20(asset).transfer(msg.sender, amount);
                emit InvestmentWithdrawn(msg.sender, asset, amount);
                break;
            }
        }
    }

    function getCurrentPrice(address asset) internal view returns (uint256) {
        // Placeholder for getting the current price of the asset
        // In a real implementation, this could call an oracle or a price feed
        return 1; // Replace with actual price fetching logic
    }

    function getWithdrawableAmount(address asset) public view returns (uint256) {
        Portfolio storage portfolio = portfolios[msg.sender];
        uint256 totalWithdrawable = 0;

        for (uint256 i = 0; i < portfolio.investments.length; i++) {
            if (portfolio.investments[i].asset == asset) {
                totalWithdrawable += portfolio.investments[i].amount;
            }
        }

        return totalWithdrawable;
    }

    function closePortfolio() public {
        Portfolio storage portfolio = portfolios[msg.sender];
        require(portfolio.owner == msg.sender, "Only the portfolio owner can close the portfolio.");

        // Withdraw all investments
        for (uint256 i = 0; i < portfolio.investments.length; i++) {
            uint256 amount = portfolio.investments[i].amount;
            if (amount > 0) {
                IERC20(portfolio.investments[i].asset).transfer(msg.sender, amount);
                emit InvestmentWithdrawn(msg.sender, portfolio.investments[i].asset, amount);
            }
        }

        delete portfolios[msg.sender]; // Clear the portfolio
        emit PortfolioClosed(msg.sender);
    }

    function getPortfolioDetails() public view returns (Investment[] memory investments, uint256 totalInvested) {
        Portfolio storage portfolio = portfolios[msg.sender];
        return (portfolio.investments, portfolio.totalInvested);
    }
}
