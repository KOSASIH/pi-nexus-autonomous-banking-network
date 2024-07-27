pragma solidity ^0.8.0;

contract KosasihUniversalisGovernanceV3 {
    address public owner;
    mapping (address => uint256) public votes;

    constructor() public {
        owner = msg.sender;
    }

    function vote(address _address, uint256 _amount) public {
        votes[_address] += _amount;
    }

    function getVotes(address _address) public view returns (uint256) {
        return votes[_address];
    }
}
