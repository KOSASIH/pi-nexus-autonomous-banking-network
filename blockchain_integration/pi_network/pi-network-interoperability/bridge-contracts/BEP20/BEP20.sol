pragma solidity ^0.8.0;

contract BEP20 {
    string public name;
    string public symbol;
    uint256 public totalSupply;

    mapping (address => uint256) public balances;

    constructor(string memory _name, string memory _symbol, uint256 _totalSupply) public {
        name = _name;
        symbol = _symbol;
        totalSupply = _totalSupply;
    }

    function transfer(address _to, uint256 _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
    }

    function getBalance(address _address) public view returns (uint256) {
        return balances[_address];
    }
}
