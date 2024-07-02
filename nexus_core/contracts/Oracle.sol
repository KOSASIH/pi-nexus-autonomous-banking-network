pragma solidity ^0.8.0;

contract Oracle {
    address public owner;
    mapping (string => uint256) public prices;

    constructor() {
        owner = msg.sender;
    }

    function setPrice(string memory symbol, uint256 price) public {
        require(msg.sender == owner, "Only the owner can set prices");
        prices[symbol] = price;
    }

    function getPrice(string memory symbol) public view returns (uint256) {
        return prices[symbol];
    }
}
