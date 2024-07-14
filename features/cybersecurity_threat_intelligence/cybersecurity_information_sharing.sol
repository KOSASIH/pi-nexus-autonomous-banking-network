// File name: cybersecurity_information_sharing.sol
pragma solidity ^0.8.0;

contract CybersecurityInformationSharing {
    mapping (address => string) public threat_intelligence;

    function share_threat_intelligence(string memory _threat_intelligence) public {
        threat_intelligence[msg.sender] = _threat_intelligence;
    }

    function get_threat_intelligence(address _address) public view returns (string memory) {
        return threat_intelligence[_address];
    }
}
