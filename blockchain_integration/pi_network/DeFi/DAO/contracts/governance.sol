pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract Governance {
    // Mapping of users to their roles
    mapping (address => uint256) public userRoles;

    // Mapping of roles to their permissions
    mapping (uint256 => uint256) public rolePermissions;

    // Event emitted when a user's role is updated
    event UserRoleUpdated(address user, uint256 role);

    // Event emitted when a role's permissions are updated
    event RolePermissionsUpdated(uint256 role, uint256 permissions);

    // Function to update a user's role
    function updateUserRole(address user, uint256 role) public {
        // Check if the user is the owner or has the required permission
        require(msg.sender == owner || userRoles[msg.sender] & rolePermissions[role] == rolePermissions[role], "Unauthorized");

        // Update the user's role
        userRoles[user] = role;

        // Emit the UserRoleUpdated event
        emit UserRoleUpdated(user, role);
    }

    // Function to update a role's permissions
    function updateRolePermissions(uint256 role, uint256 permissions) public {
        // Check if the user is the owner or has the required permission
        require(msg.sender == owner || userRoles[msg.sender] & rolePermissions[role] == rolePermissions[role], "Unauthorized");

        // Update the role's permissions
        rolePermissions[role] = permissions;

        // Emit the RolePermissionsUpdated event
        emit RolePermissionsUpdated(role, permissions);
    }
}
