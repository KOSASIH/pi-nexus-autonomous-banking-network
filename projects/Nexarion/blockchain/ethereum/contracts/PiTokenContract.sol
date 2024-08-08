pragma solidity ^0.8.0;

contract PiTokenContract {
    string public name = "Pi Token";
    string public symbol = "PI";
    uint256 public totalSupply;

    mapping (address => uint256) public balances;
    mapping (address => mapping (address => uint256)) public allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() public {
        totalSupply = 1000000000000000000000000; // 1 trillion PI tokens
        balances[msg.sender] = totalSupply;
    }

    function transfer(address to, uint256 value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    function approve(address spender, uint256 value) public {
        allowances[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
    }

    function transferFrom(address from, address to, uint256 value) public {
        require(allowances[from][msg.sender] >= value, "Insufficient allowance");
        require(balances[from] >= value, "Insufficient balance");
        balances[from] -= value;
        balances[to] += value;
        emit Transfer(from, to, value);
    }

    function balanceOf(address owner) public view returns (uint256) {
        return balances[owner];
    }

    function totalSupply() public view returns (uint256) {
        return totalSupply;
    }
}
