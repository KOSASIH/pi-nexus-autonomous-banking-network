pragma solidity ^0.8.0;

contract Compliance {
    mapping (address => bool) public compliant;

    constructor() {
        // Initialize compliance mapping
    }

    function setCompliance(address account, bool status) public {
        compliant[account] = status;
    }

    function getCompliance(address account) public view returns (bool) {
        return compliant[account];
    }
}
