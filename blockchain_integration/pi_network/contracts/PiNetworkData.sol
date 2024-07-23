pragma solidity ^0.8.0;

contract PiNetworkData {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() {
        owner = msg.sender;
    }

    function setBalance(address _address, uint256 _balance) public {
        balances[_address] = _balance;
    }

    function getBalance(address _address) public view returns (uint256) {
        return balances[_address];
    }
}
