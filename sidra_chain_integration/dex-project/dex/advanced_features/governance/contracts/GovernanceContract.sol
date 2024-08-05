pragma solidity ^0.8.0;

contract GovernanceContract {
    address public admin;
    mapping (address => bool) public voters;

    constructor() public {
        admin = msg.sender;
    }

    function propose(address _proposal) public {
        require(voters[msg.sender], "Only voters can propose");
        // ...
    }

    function vote(address _proposal) public {
        require(voters[msg.sender], "Only voters can vote");
        // ...
    }
}
