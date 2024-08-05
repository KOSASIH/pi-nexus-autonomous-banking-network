pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract RegulatoryKnowledgeGraph {
    using Roles for address;
    using SafeMath for uint256;

    // Events
    event NewRegulatoryRequirementAdded(string indexed regulatoryRequirement, uint256 timestamp);
    event RegulatoryRequirementUpdated(string indexed regulatoryRequirement, uint256 timestamp);

    // Structs
    struct RegulatoryRequirement {
        string regulatoryRequirement;
        uint256 timestamp;
        bool isActive;
    }

    // Mapping of regulatory requirements
    mapping (string => RegulatoryRequirement) public regulatoryRequirements;

    // Mapping of regulatory requirements to their corresponding compliance statuses
    mapping (string => bool) public regulatoryCompliance;

    // Constructor
    constructor() public {
        // Initialize the regulatory knowledge graph with some default regulatory requirements
        addRegulatoryRequirement("KYC", true);

    // Function to add a new regulatory requirement
    function addRegulatoryRequirement(string memory _regulatoryRequirement, bool _isActive) public onlyAdmin {
        RegulatoryRequirement storage requirement = regulatoryRequirements[_regulatoryRequirement];
        require(requirement.regulatoryRequirement == "", "Regulatory requirement already exists");
        requirement.regulatoryRequirement = _regulatoryRequirement;
        requirement.timestamp = block.timestamp;
        requirement.isActive = _isActive;
        emit NewRegulatoryRequirementAdded(_regulatoryRequirement, block.timestamp);
    }

    // Function to update a regulatory requirement
    function updateRegulatoryRequirement(string memory _regulatoryRequirement, bool _isActive) public onlyAdmin {
        RegulatoryRequirement storage requirement = regulatoryRequirements[_regulatoryRequirement];
        require(requirement.regulatoryRequirement != "", "Regulatory requirement does not exist");
        requirement.isActive = _isActive;
        emit RegulatoryRequirementUpdated(_regulatoryRequirement, block.timestamp);
    }

    // Function to get the compliance status of a regulatory requirement
    function getRegulatoryCompliance(string memory _regulatoryRequirement) public view returns (bool) {
        return regulatoryCompliance[_regulatoryRequirement];
    }

    // Modifier to restrict access to only the admin
    modifier onlyAdmin {
        require(msg.sender == admin, "Only the admin can add or update regulatory requirements");
        _;
    }

    // Admin address
    address public admin;

    // Constructor
    constructor() public {
        admin = msg.sender;
    }
}
