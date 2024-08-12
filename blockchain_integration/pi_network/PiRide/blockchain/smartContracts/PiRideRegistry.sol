pragma solidity ^0.8.0;

contract PiRideRegistry {
    address public rideFactory;
    address public userFactory;
    address public piRideToken;

    constructor() public {
        rideFactory = address(new RideFactory());
        userFactory = address(new UserFactory());
        piRideToken = address(new PiRideToken());
    }

    function getRideFactory() public view returns (address) {
        return rideFactory;
    }

    function getUserFactory() public view returns (address) {
        return userFactory;
    }

    function getPiRideToken() public view returns (address) {
        return piRideToken;
    }
}
