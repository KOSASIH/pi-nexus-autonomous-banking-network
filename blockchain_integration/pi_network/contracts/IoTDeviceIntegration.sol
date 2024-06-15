pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/security/ReentrancyGuard.sol";

contract IoTDeviceIntegration is ReentrancyGuard {
    using SafeMath for uint256;

    // Mapping of IoT devices
    mapping (address => IoTDevice) public devices;

    // Event emitted when a new IoT device is registered
    event NewIoTDevice(address indexed device, string indexed deviceType);

    // Function to register a new IoT device
    function registerIoTDevice(address device, string memory deviceType) public {
        require(!devices[device].exists, "IoT device already registered");
        devices[device] = IoTDevice(deviceType, true);
        emit NewIoTDevice(device, deviceType);
    }

    // Function to process an IoT device transaction
    function processIoTDeviceTransaction(address device, uint256 amount) public {
        require(devices[device].exists, "IoT device not registered");
        // Implement IoT device transaction processing logic
    }
}

struct IoTDevice {
    string deviceType;
    bool exists;
}
