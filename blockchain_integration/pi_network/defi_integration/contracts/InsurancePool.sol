pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract InsurancePool {
    using SafeMath for uint256;
    using SafeERC20 for ERC20;

    // Mapping of asset addresses to insurance pool balances
    mapping (address => uint256) public insurancePoolBalances;

    // Mapping of user addresses to insurance policies
    mapping (address => mapping (address => uint256)) public insurancePolicies;

    // Event emitted when an insurance policy is bought
    event InsurancePolicyBought(address user, address asset, uint256 amount);

    // Event emitted when an insurance policy is sold
    event InsurancePolicySold(address user, address asset, uint256 amount);

    // Event emitted when an insurance payout is made
    event InsurancePayout(address user, address asset, uint256 amount);

    // Function to buy an insurance policy
    function buyInsurancePolicy(address asset, uint256 amount) public {
        require(insurancePoolBalances[asset] >= amount, "Insufficient liquidity");
        require(insurancePolicies[msg.sender][asset] == 0, "Insurance policy already owned");
        insurancePoolBalances[asset] = insurancePoolBalances[asset].sub(amount);
        insurancePolicies[msg.sender][asset] = amount;
        emit InsurancePolicyBought(msg.sender, asset, amount);
    }

    // Function to sell an insurance policy
    function sellInsurancePolicy(address asset, uint256 amount) public {
        require(insurancePolicies[msg.sender][asset] == amount, "Invalid insurance policy amount");
        insurancePoolBalances[asset] = insurancePoolBalances[asset].add(amount);
        insurancePolicies[msg.sender][asset] = 0;
        emit InsurancePolicySold(msg.sender, asset, amount);
    }

    // Function to make an insurance payout
    function makeInsurancePayout(address user, address asset, uint256 amount) public {
        require(insurancePolicies[user][asset] >= amount, "Insufficient insurance policy amount");
        require(ERC20(asset).balanceOf(address(this)) >= amount, "Insufficient asset balance");
        ERC20(asset).safeTransfer(user, amount);
        insurancePolicies[user][asset] = insurancePolicies[user][asset].sub(amount);
        emit InsurancePayout(user, asset, amount);
    }
}
