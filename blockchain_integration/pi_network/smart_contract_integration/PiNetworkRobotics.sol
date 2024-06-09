pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/Robotics/Robotics.sol";

contract PiNetworkRobotics is Robotics {
    // Mapping of user addresses to their robotics devices
    mapping (address => RoboticsDevice) public roboticsDevices;

    // Struct to represent a robotics device
    struct RoboticsDevice {
        string deviceType;
        string deviceData;
    }

    // Event emitted when a new robotics device is registered
    event RoboticsDeviceRegisteredEvent(address indexed user, RoboticsDevice device);

    // Function to register a new robotics device
    function registerRoboticsDevice(string memory deviceType, string memory deviceData) public {
        RoboticsDevice storage device = roboticsDevices[msg.sender];
        device.deviceType = deviceType;
        device.deviceData = deviceData;
        emit RoboticsDeviceRegisteredEvent(msg.sender, device);
    }

    // Function to get a robotics device
    function getRoboticsDevice(address user) public view returns (RoboticsDevice memory) {
        return roboticsDevices[user];
    }
}
