pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNexusLending is SafeERC20 {
    // Lending properties
    uint256 public lendingInterest;
    uint256 public lendingPeriod;

    // Lending constructor
    constructor() public {
        lendingInterest = 5;
        lendingPeriod = 30 days;
    }

    // Lending functions
    function lend(uint256 amount) public {
        // Lend tokens to earn interest
        _lend(msg.sender, amount);
    }

    function repay() public {
        // Repay lent tokens with interest
        _repay(msg.sender);
    }

    function claimInterest() public {
        // Claim lending interest
        _claimInterest(msg.sender);
    }
}
