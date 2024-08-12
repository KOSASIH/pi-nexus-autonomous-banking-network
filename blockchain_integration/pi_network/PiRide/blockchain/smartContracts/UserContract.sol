pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract UserContract {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Mapping of user IDs to user data
    mapping (address => User) public users;

    // Event emitted when a new user is created
    event NewUser(address userId, string name, string email);

    // Event emitted when a user updates their profile
    event UserProfileUpdated(address userId, string name, string email);

    // Event emitted when a user requests a ride
    event RideRequested(address userId, uint256 rideId);

    // Event emitted when a user rates a ride
    event RideRated(address userId, uint256 rideId, uint256 rating);

    // Struct to represent a user
    struct User {
        address id;
        string name;
        string email;
        uint256 rating;
        uint256 balance;
        mapping (uint256 => RideRequest) rideRequests;
    }

    // Struct to represent a ride request
    struct RideRequest {
        uint256 rideId;
        uint256 timestamp;
        bool isAccepted;
    }

    // Function to create a new user
    function createUser(string memory name, string memory email) public {
        users[msg.sender] = User(msg.sender, name, email, 0, 0);
        emit NewUser(msg.sender, name, email);
    }

    // Function to update a user's profile
    function updateProfile(string memory name, string memory email) public {
        users[msg.sender].name = name;
        users[msg.sender].email = email;
        emit UserProfileUpdated(msg.sender, name, email);
    }

    // Function to request a ride
    function requestRide(uint256 rideId) public {
        users[msg.sender].rideRequests[rideId] = RideRequest(rideId, block.timestamp, false);
        emit RideRequested(msg.sender, rideId);
    }

    // Function to rate a ride
    function rateRide(uint256 rideId, uint256 rating) public {
        users[msg.sender].rating = users[msg.sender].rating.add(rating);
        emit RideRated(msg.sender, rideId, rating);
    }

    // Function to get a user's balance
    function getUserBalance(address user) public view returns (uint256) {
        return users[user].balance;
    }

    // Function to transfer tokens to a user
    function transferTokens(address user, uint256 amount) public {
        users[user].balance = users[user].balance.add(amount);
    }
}
