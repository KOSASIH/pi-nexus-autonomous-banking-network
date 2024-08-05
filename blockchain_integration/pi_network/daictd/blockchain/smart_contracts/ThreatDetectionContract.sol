pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract ThreatDetectionContract {
    using Roles for address;

    // Mapping of threat IDs to threat information
    mapping (uint256 => Threat) public threats;

    // Event emitted when a new threat is detected
    event NewThreatDetected(uint256 threatId, string threatInfo);

    // Event emitted when a threat is mitigated
    event ThreatMitigated(uint256 threatId);

    // Role for threat detectors
    address[] public threatDetectors;

    // Role for threat mitigators
    address[] public threatMitigators;

    // Constructor
    constructor() public {
        // Initialize roles
        threatDetectors.push(msg.sender);
        threatMitigators.push(msg.sender);
    }

    // Function to detect a new threat
    function detectThreat(uint256 threatId, string memory threatInfo) public onlyThreatDetector {
        // Create a new threat
        threats[threatId] = Threat(threatId, threatInfo);

        // Emit event
        emit NewThreatDetected(threatId, threatInfo);
    }

    // Function to mitigate a threat
    function mitigateThreat(uint256 threatId) public onlyThreatMitigator {
        // Check if threat exists
        require(threats[threatId].exists, "Threat does not exist");

        // Mitigate threat
        threats[threatId].mitigated = true;

        // Emit event
        emit ThreatMitigated(threatId);
    }

    // Modifier to restrict access to threat detectors
    modifier onlyThreatDetector {
        require(isThreatDetector(msg.sender), "Only threat detectors can call this function");
        _;
    }

    // Modifier to restrict access to threat mitigators
    modifier onlyThreatMitigator {
        require(isThreatMitigator(msg.sender), "Only threat mitigators can call this function");
        _;
    }

    // Function to check if an address is a threat detector
    function isThreatDetector(address addr) public view returns (bool) {
        return threatDetectors.includes(addr);
    }

    // Function to check if an address is a threat mitigator
    function isThreatMitigator(address addr) public view returns (bool) {
        return threatMitigators.includes(addr);
    }
}

// Struct to represent a threat
struct Threat {
    uint256 id;
    string info;
    bool mitigated;
}
