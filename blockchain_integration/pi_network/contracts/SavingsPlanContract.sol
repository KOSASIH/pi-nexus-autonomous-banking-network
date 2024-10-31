// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract SavingsPlanContract is Ownable {
    struct SavingsPlan {
        uint256 id;
        address user;
        uint256 amount;
        uint256 interestRate; // Annual interest rate in basis points (1% = 100)
        uint256 startTime;
        uint256 duration; // Duration in seconds
        bool isActive;
    }

    mapping(uint256 => SavingsPlan) public savingsPlans;
    uint256 public planCount;
    IERC20 public savingsToken;

    event SavingsPlanCreated(uint256 indexed planId, address indexed user, uint256 amount, uint256 interestRate, uint256 duration);
    event Deposited(uint256 indexed planId, uint256 amount);
    event Withdrawn(uint256 indexed planId, uint256 amount);
    event PlanClosed(uint256 indexed planId);

    constructor(address _savingsToken) {
        savingsToken = IERC20(_savingsToken);
    }

    function createSavingsPlan(uint256 amount, uint256 interestRate, uint256 duration) public returns (uint256) {
        require(amount > 0, "Amount must be greater than zero.");
        require(interestRate > 0, "Interest rate must be greater than zero.");
        require(duration > 0, "Duration must be greater than zero.");

        planCount++;
        savingsPlans[planCount] = SavingsPlan({
            id: planCount,
            user: msg.sender,
            amount: amount,
            interestRate: interestRate,
            startTime: block.timestamp,
            duration: duration,
            isActive: true
        });

        // Transfer tokens from the user to the contract
        savingsToken.transferFrom(msg.sender, address(this), amount);

        emit SavingsPlanCreated(planCount, msg.sender, amount, interestRate, duration);
        return planCount;
    }

    function deposit(uint256 planId, uint256 amount) public {
        SavingsPlan storage plan = savingsPlans[planId];
        require(plan.isActive, "Savings plan is not active.");
        require(plan.user == msg.sender, "Only the plan owner can deposit.");
        require(amount > 0, "Amount must be greater than zero.");

        // Transfer tokens from the user to the contract
        savingsToken.transferFrom(msg.sender, address(this), amount);
        plan.amount += amount;

        emit Deposited(planId, amount);
    }

    function withdraw(uint256 planId) public {
        SavingsPlan storage plan = savingsPlans[planId];
        require(plan.isActive, "Savings plan is not active.");
        require(plan.user == msg.sender, "Only the plan owner can withdraw.");
        require(block.timestamp >= plan.startTime + plan.duration, "Savings plan duration has not ended.");

        uint256 totalAmount = plan.amount + calculateInterest(plan);
        plan.isActive = false; // Mark the plan as inactive

        // Transfer the total amount back to the user
        savingsToken.transfer(msg.sender, totalAmount);

        emit Withdrawn(planId, totalAmount);
        emit PlanClosed(planId);
    }

    function calculateInterest(SavingsPlan memory plan) internal view returns (uint256) {
        uint256 timeElapsed = block.timestamp - plan.startTime;
        uint256 interest = (plan.amount * plan.interestRate * timeElapsed) / (365 days * 10000); // Basis points to percentage
        return interest;
    }

    function getPlanDetails(uint256 planId) public view returns (address user, uint256 amount, uint256 interestRate, uint256 startTime, uint256 duration, bool isActive) {
        SavingsPlan memory plan = savingsPlans[planId];
        return (plan.user, plan.amount, plan.interestRate, plan.startTime, plan.duration, plan.isActive);
    }
}
