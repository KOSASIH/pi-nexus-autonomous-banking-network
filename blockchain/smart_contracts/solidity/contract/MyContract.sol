pragma solidity ^0.8.0;

contract MyContract {
    uint256 public storedData;

    constructor() {
        storedData = 0;
    }

    function set(uint256 data) public {
        storedData = data;
    }

    function get() public view returns (uint256) {
        return storedData;
    }
}
