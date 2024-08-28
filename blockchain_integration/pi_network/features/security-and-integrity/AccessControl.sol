pragma solidity ^0.8.0;

contract AccessControl {
    // Mapping of access control roles
    mapping (address => Role) public roles;

    // Struct to represent an access control role
    struct Role {
        address owner;
        bytes32 role;
        bool authorized;
    }

    // Event emitted when a new role is created
    event NewRole(address indexed owner, bytes32 role);

    // Event emitted when a role is updated
    event UpdateRole(address indexed owner, bytes32 role);

    // Function to create a new role
    function createRole(bytes32 _role) public {
        address owner = msg.sender;
        Role storage role = roles[owner];
        role.owner = owner;
        role.role = _role;
        role.authorized = true;
        emit NewRole(owner, _role);
    }

    // Function to update a role
    function updateRole(bytes32 _role) public {
        address owner = msg.sender;
        Role storage role = roles[owner];
        require(role.owner == owner, "Unauthorized access");
        role.role = _role;
        emit UpdateRole(owner, _role);
    }

    // Function to check if a user has a specific role
    function hasRole(address _owner, bytes32 _role) internal returns (bool) {
        Role storage role = roles[_owner];
        return role.role == _role && role.authorized;
    }
}
