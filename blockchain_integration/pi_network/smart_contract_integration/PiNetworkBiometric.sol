pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/Biometric/Biometric.sol";

contract PiNetworkBiometric is Biometric {
    // Mapping of user addresses to their biometric data
    mapping (address => BiometricData) public biometricData;

    // Struct to represent biometric data
    struct BiometricData {
        string dataType;
        string dataValue;
    }

    // Event emitted when new biometric data is received
    event BiometricDataReceivedEvent(address indexed user, BiometricData data);

    // Function to receive biometric data
    function receiveBiometricData(string memory dataType, string memory dataValue) public {
        BiometricData storage data = biometricData[msg.sender];
        data.dataType = dataType;
        data.dataValue = dataValue;
        emit BiometricDataReceivedEvent(msg.sender, data);
    }

    // Function to get biometric data
    function getBiometricData(address user) public view returns (BiometricData memory) {
        return biometricData[user];
    }
}
