pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedCrowdfunding {
    // Mapping of investor addresses to investment amounts
    mapping (address => uint256) public investments;

    // Event emitted when a new investment is made
    event InvestmentMade(address investor, uint256 amount);

    // Function to make a new investment
    function makeInvestment(uint256 _amount) public {
        // Check if investment amount is valid
        require(_amount > 0, "Invalid investment amount");

        // Add investment to mapping
        investments[msg.sender] = investments[msg.sender].add(_amount);

        // Emit investment made event
        emit InvestmentMade(msg.sender, _amount);
    }

    // Function to withdraw investments
    function withdrawInvestments() public {
        // Withdraw investment amount to investor
        uint256 investmentAmount = investments[msg.sender];
        investments[msg.sender] = 0;

        // Transfer funds to investor
        msg.sender.transfer(investmentAmount);
    }
}
