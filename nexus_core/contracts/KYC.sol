pragma solidity ^0.8.0;

contract KYC {
    mapping (address => string) public kycData;

    constructor() {
        // Initialize KYC data mapping
    }

    function setKYCData(string memory data) public {
        kycData[msg.sender] = data;
    }

    function getKYCData(address account) public view returns (string memory) {
        return kycData[account];
    }
}
