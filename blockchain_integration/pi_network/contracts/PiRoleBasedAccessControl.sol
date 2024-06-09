pragma solidity ^0.8.0;

contract PiRoleBasedAccessControl {
    mapping (address => Role) public roles;

    enum Role {
        NONE,
        OWNER,
        ADMIN,
        USER
    }

    constructor() public {
        roles[msg.sender] = Role.OWNER;
    }

    function setRole(address _address, Role _role) public {
        require(roles[msg.sender] == Role.OWNER, "Only the owner can set roles");
        roles[_address] = _role;
    }

    function hasRole(address _address, Role _role) public view returns (bool) {
        return roles[_address] == _role;
    }

    modifier onlyRole(Role _role) {
        require(hasRole(msg.sender, _role), "Access denied");
        _;
    }
}
