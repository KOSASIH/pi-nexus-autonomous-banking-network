pragma solidity ^0.8.0;

contract NexusBlockchainSmartContract {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function transferFunds(address recipient, uint256 amount) public {
        require(msg.sender == owner, "Only the owner can transfer funds");
        balances[recipient] += amount;
    }

    function getBalance(address account) public view returns (uint256) {
        return balances[account];
    }
}
