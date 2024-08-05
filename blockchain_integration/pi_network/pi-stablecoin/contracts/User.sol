pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract User {
    using Roles for address;

    // Mapping of user roles
    mapping (address => bool) public isAdmin;

    // Event emitted when a user is added
    event UserAdded(address indexed user);

    // Event emitted when a user is removed
    event UserRemoved(address indexed user);

    // Function to add a user
    function addUser(address user) public {
        // Only allow adding users by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can add users");

        // Add user
        isAdmin[user] = true;
        emit UserAdded(user);
    }

    // Function to remove a user
    function removeUser(address user) public {
        // Only allow removing users by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can remove users");

        // Remove user
        isAdmin[user] = false;
        emit UserRemoved(user);
    }
}
