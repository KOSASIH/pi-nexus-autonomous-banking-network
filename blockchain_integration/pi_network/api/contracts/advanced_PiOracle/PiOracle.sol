// PiOracle.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiOracle {
    //...
    function updateData(bytes32 dataFeed, uint256 value) public onlyOracle {
        //...
    }

    function validateData(bytes32 dataFeed, uint256 value) public view returns (bool) {
        //...
    }

    function getData(bytes32 dataFeed) public view returns (uint256) {
        //...
    }

    //...
}
