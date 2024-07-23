pragma solidity ^0.8.0;

contract PiNetworkOracle {
    address private owner;
    mapping (address => uint256) public prices;

    constructor() {
        owner = msg.sender;
    }

    function setPrice(address _address, uint256 _price) public {
        prices[_address] = _price;
    }

    function getPrice(address _address) public view returns (uint256) {
        return prices[_address];
    }
}
