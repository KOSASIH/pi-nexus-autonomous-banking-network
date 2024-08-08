// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract Lending is ReentrancyGuard {
    struct LendingPool {
        address payable owner;
        address asset;
        uint256 interestRate;
        uint256 totalDeposited;
        uint256 totalBorrowed;
        mapping(address => uint256) balances;
    }

    LendingPool[] public lendingPools;

    function createLendingPool(address asset, uint256 interestRate) public {
        require(asset != address(0), "Invalid asset address");
        require(interestRate > 0, "Invalid interest rate");

        LendingPool memory pool;
        pool.owner = payable(msg.sender);
        pool.asset = asset;
        pool.interestRate = interestRate;
        pool.totalDeposited = 0;
        pool.totalBorrowed = 0;

        lendingPools.push(pool);
    }

    function deposit(uint256 index) public payable {
        require(index > 0, "Invalid index");

        LendingPool storage pool = lendingPools[index];
        require(pool.asset == address(this), "Invalid lending pool address");

        pool.balances[msg.sender] += msg.value;
        pool.totalDeposited += msg.value;
    }

    function borrow(uint256 index, uint256 amount) public {
        require(index > 0, "Invalid index");

        LendingPool storage pool = lendingPools[index];
        require(pool.asset == address(this), "Invalid lending pool address");
        require(amount > 0, "Invalid amount");

        uint256 interest = (amount * pool.interestRate) / 100;
        require(pool.balances[msg.sender] >= interest, "Insufficient balance");

        pool.balances[msg.sender] -= interest;
        pool.totalBorrowed += interest;
    }

    function repay(uint256 index, uint256 amount) public {
        require(index > 0, "Invalid index");

        LendingPool storage pool = lendingPools[index];
        require(pool.asset == address(this), "Invalid lending pool address");
        require(amount > 0, "Invalid amount");

        pool.balances[msg.sender] += amount;
        pool.totalBorrowed -= amount;
    }

    function getBalance(uint256 index, address user) public view returns (uint256) {
        LendingPool storage pool = lendingPools[index];
        return pool.balances[user];
    }

    function getTotalDeposited(uint256 index) public view returns (uint256) {
        LendingPool storage pool = lendingPools[index];
        return pool.totalDeposited;
    }

    function getTotalBorrowed(uint256 index) public view returns (uint256) {
        LendingPool storage pool = lendingPools[index];
        return pool.totalBorrowed;
    }
}
