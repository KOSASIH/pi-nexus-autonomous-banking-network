pragma solidity ^0.8.0;

contract InsuranceContract {
    address private owner;
    mapping (address => uint256) public insuredAmounts;

    constructor() public {
        owner = msg.sender;
    }

    function insure(address insured, uint256 amount) public {
        insuredAmounts[insured] = amount;
    }

    function claim(address insured, uint256 amount) public {
        require(insuredAmounts[insured] >= amount, 'Insufficient insured amount');
        insuredAmounts[insured] -= amount;
        // Pay out claim
    }
}
