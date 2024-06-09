pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/IoT/IoT.sol";

contract PiNetworkIoT is IoT {
    // Mapping of device addresses to their corresponding data
    mapping (address => IoTData) public deviceData;

    // Struct to represent IoT data
    struct IoTData {
        string dataType;
        string dataValue;
    }

    // Event emitted when new IoT data is received
    event IoTDataReceivedEvent(address indexed device, IoTData data);

    // Function to receive IoT data
    function receiveIoTData(string memory dataType, string memory dataValue) public {
        IoTData storage data = deviceData[msg.sender];
        data.dataType = dataType;
        data.dataValue = dataValue;
        emit IoTDataReceivedEvent(msg.sender, data);
    }

    // Function to get IoT data
    function getIoTData(address device) public view returns (IoTData memory) {
        return deviceData[device];
    }
}
