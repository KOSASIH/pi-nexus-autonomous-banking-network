pragma solidity ^0.8.0;

contract ExternalBlockchainExample is IExternalBlockchain {
    // Event emitted when data is received
    event DataReceived(address indexed from, bytes data);

    // Function to send data to the Pi Nexus blockchain
    function sendData(address _to, bytes calldata _data) external override returns (bool) {
        // Emit an event to indicate data has been sent
        emit DataReceived(msg.sender, _data);
        return true;
    }

    // Function to receive data from the Pi Nexus blockchain
    function receiveData(address _from, bytes calldata _data) external override {
        // Handle the received data (e.g., store it, process it, etc.)
        emit DataReceived(_from, _data);
    }
}
