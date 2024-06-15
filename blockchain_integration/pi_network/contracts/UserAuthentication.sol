pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract UserAuthentication is Ownable {
    // Mapping of user credentials
    mapping (address => User) public users;

    // Event emitted when a user is authenticated
    event UserAuthenticated(address indexed user);

    // Function to authenticate a user
    function authenticateUser(address user, string memory password) public {
        require(users[user].password == password, "Invalid password");
        emit UserAuthenticated(user);
    }

    // Function to register a new user
    function registerUser(address user, string memory password) public {
        require(users[user] == 0, "User already exists");
        users[user] = User(password, 0);
    }

    // Function to update user credentials
    function updateUserCredentials(address user, string memory newPassword) public {
        require(users[user] != 0, "User does not exist");
        users[user].password = newPassword;
    }
}

struct User {
    string password;
    uint256 authenticationCount;
}
