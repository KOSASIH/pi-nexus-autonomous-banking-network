pragma solidity ^0.8.0;

import "./GovernanceModel.sol";

contract GovernanceController {
    // Governance model instance
    GovernanceModel public governanceModel;

    // Event emitted when a user's role is updated
    event UserRoleUpdated(address user, uint256 role);

    // Event emitted when a role's permissions are updated
    event RolePermissionsUpdated(uint256 role, uint256 permissions);

    // Constructor
    constructor(address governanceModelAddress) public {
        governanceModel = GovernanceModel(governanceModelAddress);
    }

    // Function to update a user's role
    function updateUserRole(address user, uint256 role) public {
        governanceModel.updateUserRole(user, role);
        emit UserRoleUpdated(user, role);
    }

    // Function to update a role's permissions
    function updateRolePermissions(uint256 role, uint256 permissions) public {
        governanceModel.updateRolePermissions(role, permissions);
        emit RolePermissionsUpdated(role, permissions);
    }

    // Function to get a user's role
    function getUserRole(address user) public view returns (uint256) {
        return governanceModel.getUserRole(user);
    }

    // Function to get a role's permissions
    function getRolePermissions(uint256 role) public view returns (uint256) {
        return governanceModel.getRolePermissions(role);
    }
}
