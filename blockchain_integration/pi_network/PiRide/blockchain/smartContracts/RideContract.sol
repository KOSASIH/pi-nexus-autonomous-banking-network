pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract RideContract {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Mapping of ride IDs to ride data
    mapping (uint256 => Ride) public rides;

    // Mapping of user IDs to user data
    mapping (address => User) public users;

    // Event emitted when a new ride is created
    event NewRide(uint256 rideId, address creator, uint256 departureTime, uint256 arrivalTime, uint256 price);

    // Event emitted when a user requests a ride
    event RideRequested(uint256 rideId, address requester);

    // Event emitted when a ride is accepted
    event RideAccepted(uint256 rideId, address driver);

    // Event emitted when a ride is completed
    event RideCompleted(uint256 rideId, address driver, uint256 rating);

    // Struct to represent a ride
    struct Ride {
        uint256 id;
        address creator;
        uint256 departureTime;
        uint256 arrivalTime;
        uint256 price;
        address driver;
        uint256 rating;
        bool isAvailable;
    }

    // Struct to represent a user
    struct User {
        address id;
        string name;
        string email;
        uint256 rating;
        uint256 balance;
    }

    // Function to create a new ride
    function createRide(uint256 departureTime, uint256 arrivalTime, uint256 price) public {
        uint256 rideId = uint256(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
        rides[rideId] = Ride(rideId, msg.sender, departureTime, arrivalTime, price, address(0), 0, true);
        emit NewRide(rideId, msg.sender, departureTime, arrivalTime, price);
    }

    // Function to request a ride
    function requestRide(uint256 rideId) public {
        require(rides[rideId].isAvailable, "Ride is not available");
        rides[rideId].isAvailable = false;
        emit RideRequested(rideId, msg.sender);
    }

    // Function to accept a ride
    function acceptRide(uint256 rideId) public {
        require(rides[rideId].isAvailable == false, "Ride is not available");
        rides[rideId].driver = msg.sender;
        emit RideAccepted(rideId, msg.sender);
    }

    // Function to complete a ride
    function completeRide(uint256 rideId, uint256 rating) public {
        require(rides[rideId].driver == msg.sender, "Only the driver can complete the ride");
        rides[rideId].rating = rating;
        emit RideCompleted(rideId, msg.sender, rating);
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
