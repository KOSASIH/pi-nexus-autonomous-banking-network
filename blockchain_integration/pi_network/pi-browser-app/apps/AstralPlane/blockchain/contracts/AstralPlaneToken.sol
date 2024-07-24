// AstralPlaneToken.sol
pragma solidity ^0.8.0;

contract AstralPlaneToken {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
        balances[owner] = 1000000;
    }

    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }
}
