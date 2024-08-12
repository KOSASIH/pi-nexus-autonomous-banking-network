pragma solidity ^0.8.0;

import "./RideContract.sol";

contract RideFactory {
    address[] public rideContracts;

    event NewRideContract(address rideContract);

    function createRideContract() public {
        RideContract rideContract = new RideContract();
        rideContracts.push(address(rideContract));
        emit NewRideContract(address(rideContract));
    }

    function getRideContracts() public view returns (address[] memory) {
        return rideContracts;
    }
}
