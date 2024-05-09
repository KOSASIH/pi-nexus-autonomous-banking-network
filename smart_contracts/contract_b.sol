pragma solidity ^0.8.0;

import "./contract_a.sol";

contract ContractB is ContractA {
    uint256 public storedData2;

    constructor() {
        storedData2 = 0;
    }

    function set2(uint256 data) public {
        storedData2 = data;
    }

    function get2() public view returns (uint256) {
        return storedData2;
    }
}
