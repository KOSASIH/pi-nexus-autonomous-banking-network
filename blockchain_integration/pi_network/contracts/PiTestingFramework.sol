pragma solidity ^0.8.0;

contract PiTestingFramework {
    mapping (address => bool) public testedContracts;

    function testContract(address _contract) public {
        // Implement testing logic here
        testedContracts[_contract] = true;
    }

    function isContractTested(address _contract) public view returns (bool) {
        return testedContracts[_contract];
    }
}
