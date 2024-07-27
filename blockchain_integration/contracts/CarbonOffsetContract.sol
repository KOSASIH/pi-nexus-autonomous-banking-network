pragma solidity ^0.8.0;

contract CarbonOffsetContract {
    // Mapping of project IDs to project details
    mapping(uint256 => Project) public projects;

    // Mapping of user addresses to their carbon offset balances
    mapping(address => uint256) public carbonOffsetBalances;

    // Event emitted when a user invests in a project
    event Invested(uint256 projectId, address user, uint256 amount);

    // Event emitted when a user's carbon offset balance is updated
    event CarbonOffsetUpdated(address user, uint256 newBalance);

    // Struct to represent a project
    struct Project {
        string name;
        uint256 carbonOffsetPerPi;
        uint256 totalInvested;
        uint256 projectId;
    }

    // Function to add a new project
    function addProject(string memory _name, uint256 _carbonOffsetPerPi) public {
        Project memory newProject = Project(_name, _carbonOffsetPerPi, 0, projects.length);
        projects[projects.length] = newProject;
    }

    // Function to invest in a project
    function invest(uint256 _projectId, uint256 _amount) public {
        Project storage project = projects[_projectId];
        require(project.projectId != 0, "Project does not exist");
        uint256 carbonOffset = _amount * project.carbonOffsetPerPi;
        carbonOffsetBalances[msg.sender] += carbonOffset;
        project.totalInvested += _amount;
        emit Invested(_projectId, msg.sender, _amount);
        emit CarbonOffsetUpdated(msg.sender, carbonOffsetBalances[msg.sender]);
    }

    // Function to view a user's carbon offset balance
    function viewCarbonOffsetBalance() public view returns (uint256) {
        return carbonOffsetBalances[msg.sender];
    }

    // Function to view a project's details
    function viewProject(uint256 _projectId) public view returns (Project memory) {
        return projects[_projectId];
    }
}
