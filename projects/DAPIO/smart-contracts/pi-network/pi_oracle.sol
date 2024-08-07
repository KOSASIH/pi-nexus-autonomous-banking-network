pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/oracle/Oracle.sol";

contract PiOracle is Oracle {
    mapping (address => uint256) public prices;

    constructor() public {
        // Initialize prices with some default values
        prices[0x0000000000000000000000000000000000000001] = 100; // USD
        prices[0x0000000000000000000000000000000000000002] = 200; // EUR
        prices[0x0000000000000000000000000000000000000003] = 300; // JPY
    }

    function getPrice(address _token) public view returns (uint256) {
        return prices[_token];
    }

    function updatePrice(address _token, uint256 _price) public onlyOwner {
        prices[_token] = _price;
    }
}
