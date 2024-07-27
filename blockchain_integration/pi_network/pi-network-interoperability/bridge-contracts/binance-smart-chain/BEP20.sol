pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract BEP20 {
    string public name;
    string public symbol;
    uint256 public totalSupply;
    uint256 public decimals;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol, uint256 _totalSupply, uint256 _decimals) public {
        name = _name;
        symbol = _symbol;
        totalSupply = _totalSupply;
        decimals = _decimals;
    }

    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
    }

    function approve(address spender, uint256 amount) public {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public {
        require(allowances[sender][msg.sender] >= amount, "Insufficient allowance");
        require(balances[sender] >= amount, "Insufficient balance");
        balances[sender] -= amount;
        balances[recipient] += amount;
        allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return allowances[owner][spender];
    }
}
