// SPDX-License-Identifier: MIT
    pragma solidity ^0.8.20;
    
    import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
    import "@openzeppelin/contracts/access/Ownable.sol";
    
    contract ImpactInvestingContract is Ownable {
        struct Project {
            uint256 id;
            string name;
            string description;
            uint256 goal;
            uint256 raised;
            bool active;
        }
    
        struct Investment {
            uint256 projectId;
            uint256 amount;
        }
    
        mapping(uint256 => Project) public projects;
        mapping(address => mapping(uint256 => Investment)) public investments;
        mapping(address => uint256[]) public userInvestedProjects;
    
        event ProjectAdded(uint256 indexed projectId, string projectName, uint256 goal);
        event InvestmentMade(address indexed user, uint256 indexed projectId, uint256 amount);
    
        constructor() Ownable(msg.sender) {}
    
        function addProject(string memory _name, string memory _description, uint256 _goal) external onlyOwner {
            uint256 projectId = uint256(keccak256(abi.encodePacked(_name, _description, _goal, block.timestamp)));
            projects[projectId] = Project(projectId, _name, _description, _goal, 0, true);
            emit ProjectAdded(projectId, _name, _goal);
        }
    
        function invest(uint256 _projectId, uint256 _amount) external payable {
            Project storage proj = projects[_projectId];
            require(proj.active, "Project does not exist or is inactive");
            require(_amount > 0, "Investment amount must be greater than zero");
    
            proj.raised += _amount;
            
            if (investments[msg.sender][_projectId].amount == 0) {
                userInvestedProjects[msg.sender].push(_projectId);
            }
            
            investments[msg.sender][_projectId] = Investment(_projectId, investments[msg.sender][_projectId].amount + _amount);
    
            emit InvestmentMade(msg.sender, _projectId, _amount);
        }
    
        function getProject(uint256 _projectId) external view returns (Project memory) {
            return projects[_projectId];
        }
    
        function getUserInvestments(address _user) external view returns (Investment[] memory) {
            uint256[] memory projIds = userInvestedProjects[_user];
            Investment[] memory userInvestments = new Investment[](projIds.length);
            for (uint256 i = 0; i < projIds.length; i++) {
                userInvestments[i] = investments[_user][projIds[i]];
            }
            return userInvestments;
        }
    }
    