pragma solidity ^0.8.0;

contract PiNetworkContract {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function deposit(address _address, uint256 _amount) public {
        balances[_address] += _amount;
    }

    function withdraw(address _address, uint256 _amount) public {
        require(balances[_address] >= _amount, "Insufficient balance");
        balances[_address] -= _amount;
    }

    function getBalance(address _address) public view returns (uint256) {
        return balances[_address];
    }
}
