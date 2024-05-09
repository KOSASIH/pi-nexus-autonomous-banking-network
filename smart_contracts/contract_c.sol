pragma solidity ^0.8.0;

import "./contract_a.sol";

contract ContractC {
    ContractA public contractA;

    constructor(address contractAAddress) {
        contractA = ContractA(contractAAddress);
    }

    function setContractAData(uint256 data) public {
        contractA.set(data);
    }

    function getContractAData() public view returns (uint256) {
        return contractA.get();
    }
}
