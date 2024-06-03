pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PI_Nexus_Token is ERC20 {
    address private owner;
    uint256 public totalSupply;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
        totalSupply = 100000000 * (10 ** 18); // 100 million tokens
        balances[owner] = totalSupply;
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(allowances[sender][msg.sender] >= amount, "Insufficient allowance");
        require(balances[sender] >= amount, "Insufficient balance");
        balances[sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }
}
