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
    function invest(uint256 _projectId, uint256 _amount) public {
        Project storage project = projects[_projectId];
        require(project.projectId != 0, "Project does not exist");
        uint256 impact = _amount * project.impactPerPi;
        impactBalances[msg.sender] += impact;
        project.totalInvested += _amount;
        emit Invested(_projectId, msg.sender, _amount);
        emit ImpactUpdated(msg.sender, impactBalances[msg.sender]);
    }

    // Function to view a user's impact balance
    function viewImpactBalance() public view returns (uint256) {
        return impactBalances[msg.sender];
    }

    // Function to view a project's details
    function viewProject(uint256 _projectId) public view returns (Project memory) {
        return projects[_projectId];
    }
}
