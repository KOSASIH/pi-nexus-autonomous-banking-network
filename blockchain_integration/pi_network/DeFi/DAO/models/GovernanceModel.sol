pragma solidity ^0.8.0;

contract GovernanceModel {
    // Mapping of roles
    mapping (uint256 => Role) public roles;

    // Mapping of users
    mapping (address => User) public users;

    // Event emitted when a user's role is updated
    event UserRoleUpdated(address user, uint256 role);

    // Event emitted when a role's permissions are updated
    event RolePermissionsUpdated(uint256 role, uint256 permissions);

    // Struct to represent a role
    struct Role {
        uint256 id;
        uint256 permissions;
    }

    // Struct to represent a user
    struct User {
        address addr;
        uint256 role;
    }

    // Constructor
    constructor() public {}

    // Function to update a user's role
    function updateUserRole(address user, uint256 role) public {
        users[user].role = role;
        emit UserRoleUpdated(user, role);
    }

    // Function to update a role's permissions
    function updateRolePermissions(uint256 role, uint256 permissions) public {
        roles[role].permissions = permissions;
        emit RolePermissionsUpdated(role, permissions);
    }

    // Function to get a user's role
    function getUserRole(address user) public view returns (uint256) {
        return users[user].role;
    }

    // Function to get a role's permissions
    function getRolePermissions(uint256 role) public view returns (uint256) {
        return roles[role].permissions;
    }
}
