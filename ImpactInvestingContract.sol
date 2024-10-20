pragma solidity ^0.8 .0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract ImpactInvestingContract {
    // Mapping of project IDs to project details
    mapping(uint => Project) public projects;

    // Mapping of user addresses to their investments
    mapping(address => mapping(uint => Investment)) public investments;

    // Event emitted when a new project is added
    event ProjectAdded(uint projectId, string projectName, string projectDescription);

    // Event emitted when a user invests in a project
    event InvestmentMade(address indexed user, uint projectId, uint amount);

    // Struct to represent a project
    struct Project {
        uint id;
        string name;
        string description;
        uint goal;
        uint raised;
    }

    // Struct to represent an investment
    struct Investment {
        uint projectId;
        uint amount;
    }

    // Function to add a new project
    function addProject(string memory _name, string memory _description, uint _goal) public {
        // Generate a unique project ID
        uint projectId = uint(keccak256(abi.encodePacked(_name, _description, _goal)));

        // Create a new project
        projects[projectId] = Project(projectId, _name, _description, _goal, 0);

        // Emit the ProjectAdded event
        emit ProjectAdded(projectId, _name, _description);
    }

    // Function to invest in a project
    function invest(uint _projectId, uint _amount) public {
        // Check if the project exists
        require(projects[_projectId].id != 0, "Project does not exist");

        // Check if the user has already invested in the project
        require(investments[msg.sender][_projectId].amount == 0, "User has already invested in this project");

        // Update the project's raised amount
        projects[_projectId].raised += _amount;

        // Create a new investment
        investments[msg.sender][_projectId] = Investment(_projectId, _amount);

        // Emit the InvestmentMade event
        emit InvestmentMade(msg.sender, _projectId, _amount);
    }

    // Function to get a project's details
    function getProject(uint _projectId) public view returns (Project memory) {
        return projects[_projectId];
    }

    // Function to get a user's investments
    function getInvestments(address _user) public view returns (Investment[] memory) {
        Investment[] memory userInvestments = new Investment[](projects.length);

        for (uint i = 0; i < projects.length; i++) {
            if (investments[_user][i].amount != 0) {
                userInvestments.push(investments[_user][i]);
            }
        }

        return userInvestments;
    }
}
