pragma solidity ^0.8.0;

import "./PiToken.sol";

contract PiNexus {
    address public piTokenAddress;
    mapping (address => uint256) public userBalances;

    constructor(address _piTokenAddress) public {
        piTokenAddress = _piTokenAddress;
    }

    function deposit(uint256 _amount) public {
        //...
    }

    function withdraw(uint256 _amount) public {
        //...
    }

    function transfer(address _to, uint256 _amount) public {
        //...
    }
}
