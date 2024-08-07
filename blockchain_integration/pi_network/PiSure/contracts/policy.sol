pragma solidity ^0.8.0;

contract Policy {
    address public policyHolder;
    uint public amount;
    uint public premium;

    constructor(address _policyHolder, uint _amount) {
        policyHolder = _policyHolder;
        amount = _amount;
        premium = calculatePremium(_amount);
    }

    function calculatePremium(uint _amount) internal pure returns (uint) {
        // Premium calculation logic goes here
        return _amount * 0.05;
    }
}
