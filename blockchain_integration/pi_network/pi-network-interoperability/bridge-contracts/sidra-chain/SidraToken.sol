pragma solidity ^0.8.0;

contract SidraToken {
    string public name;
    string public symbol;
    uint256 public totalSupply;
    mapping(address => uint256) public balances;

    event Transfer(address indexed from, address indexed to, uint256 value);

    constructor() public {
        name = "Sidra Token";
        symbol = "ST";
        totalSupply = 100000000 * (10**18); // 100 million tokens
        balances[msg.sender] = totalSupply;
    }

    function transfer(address _to, uint256 _value) public {
        require(balances[msg.sender] >= _value, "Insufficient balance");
        balances[msg.sender] -= _value;
        balances[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
    }

    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
}
