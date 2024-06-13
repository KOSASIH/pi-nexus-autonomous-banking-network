pragma solidity ^0.8.0;

library GovernanceManager {
    // Function to update a user's role
    function updateUserRole(address user, uint256 role) public {
        // Get the governance contract
        Governance governance = Governance(address);

        // Update the user's role
        governance.updateUserRole(user, role);
    }

    // Function to update a role's permissions
    function updateRolePermissions(uint256 role, uint256 permissions) public {
        // Get the governance contract
        Governance governance = Governance(address);

        // Update the role's permissions
        governance.updateRolePermissions(role, permissions);
    }
}
