pragma solidity ^0.8.0;

contract SustainableDevelopmentContract {
    // Mapping of project IDs to project details
    mapping(uint256 => Project) public projects;

    // Mapping of user addresses to their impact balances
    mapping(address => uint256) public impactBalances;

    // Event emitted when a user invests in a project
    event Invested(uint256 projectId, address user, uint256 amount);

    // Event emitted when a user's impact balance is updated
    event ImpactUpdated(address user, uint256 newBalance);

    // Struct to represent a project
    struct Project {
        string name;
        uint256 impactPerPi;
        uint256 totalInvested;
        uint256 projectId;
    }

    // Function to add a new project
    function addProject(string memory _name, uint256 _impactPerPi) public {
        Project memory newProject = Project(_name, _impactPerPi, 0, projects.length);
        projects[projects.length] = newProject;
    }

    // Function to invest in a project
    function invest(uint256 _projectId, uint256 _amount
