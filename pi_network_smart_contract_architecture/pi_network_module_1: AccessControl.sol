pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract AccessControl {
    using Roles for address;

    // Mapping of roles to addresses
    mapping (address => Role) public roles;

    // Enum for roles
    enum Role { ADMIN, DEVELOPER, USER }

    // Modifier to restrict access to admins
    modifier onlyAdmin() {
        require(roles[msg.sender] == Role.ADMIN, "Only admins can access this function");
        _;
    }

    // Modifier to restrict access to developers
    modifier onlyDeveloper() {
        require(roles[msg.sender] == Role.DEVELOPER, "Only developers can access this function");
        _;
    }

    // Function to add a new role
    function addRole(address _address, Role _role) public onlyAdmin {
        roles[_address] = _role;
    }

    // Function to remove a role
    function removeRole(address _address) public onlyAdmin {
        delete roles[_address];
    }
}
