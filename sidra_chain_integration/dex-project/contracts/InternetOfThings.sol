pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract InternetOfThings is Ownable, ReentrancyGuard {
    // Mapping of devices to their respective owners
    mapping(address => address) public deviceOwners;

    // Mapping of devices to their respective data
    mapping(address => bytes) public deviceData;

    // Event emitted when a new device is registered
    event NewDevice(address indexed device, address indexed owner);

    // Event emitted when a device is updated
    event UpdateDevice(address indexed device, bytes data);

    // Event emitted when a device is removed
    event RemoveDevice(address indexed device);

    // Function to register a new device
    function registerDevice(address device) public {
        // Check if the device is not already registered
        require(deviceOwners[device] == address(0), "Device already registered");

        // Set the owner of the device
        deviceOwners[device] = msg.sender;

        // Emit the NewDevice event
        emit NewDevice(device, msg.sender);
    }

    // Function to update a device
    function updateDevice(address device, bytes memory data) public {
        // Check if the device is registered
        require(deviceOwners[device] != address(0), "Device not registered");

        // Check if the caller is the owner of the device
        require(deviceOwners[device] == msg.sender, "Only the owner can update the device");

        // Update the device data
        deviceData[device] = data;

        // Emit the UpdateDevice event
        emit UpdateDevice(device, data);
    }

    // Function to remove a device
    function removeDevice(address device) public {
        // Check if the device is registered
        require(deviceOwners[device] != address(0), "Device not registered");

        // Check if the caller is the owner of the device
        require(deviceOwners[device] == msg.sender, "Only the owner can remove the device");

        // Remove the device
        delete deviceOwners[device];
        delete deviceData[device];

        // Emit the RemoveDevice event
        emit RemoveDevice(device);
    }

    // Function to get the data of a device
    function getDeviceData(address device) public view returns (bytes memory) {
        // Check if the device is registered
        require(deviceOwners[device] != address(0), "Device not registered");

        // Return the device data
        return deviceData[device];
    }
}
