pragma solidity ^0.8.0;

interface IGovernance {
    // Function to update a user's role
    function updateUserRole(address user, uint256 role) external;

    // Function to update a role's permissions
    function updateRolePermissions(uint256 role, uint256 permissions) external;

    // Function to get a user's role
    function getUserRole(address user) external view returns (uint256);

    // Function to get a role's permissions
    function getRolePermissions(uint256 role) external view returns (uint256);

    // Event emitted when a user's role is updated
    event UserRoleUpdated(address user, uint256 role);

    // Event emitted when a role's permissions are updated
    event RolePermissionsUpdated(uint256 role, uint256 permissions);
}
