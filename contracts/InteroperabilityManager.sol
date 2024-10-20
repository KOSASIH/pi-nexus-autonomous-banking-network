pragma solidity ^0.8.0;

interface IExternalBlockchain {
    function sendData(address _to, bytes calldata _data) external returns (bool);
    function receiveData(address _from, bytes calldata _data) external;
}

contract InteroperabilityManager {
    // Define the mapping of external blockchain addresses
    mapping(string => address) public externalBlockchains;

    // Event emitted when a new external blockchain is registered
    event ExternalBlockchainRegistered(string indexed name, address indexed blockchainAddress);

    // Function to register an external blockchain
    function registerExternalBlockchain(string memory _name, address _blockchainAddress) public {
        require(_blockchainAddress != address(0), "Invalid address");
        externalBlockchains[_name] = _blockchainAddress;
        emit ExternalBlockchainRegistered(_name, _blockchainAddress);
    }

    // Function to send data to an external blockchain
    function sendDataToExternalBlockchain(string memory _name, bytes calldata _data) public returns (bool) {
        address blockchainAddress = externalBlockchains[_name];
        require(blockchainAddress != address(0), "Blockchain not registered");

        IExternalBlockchain externalBlockchain = IExternalBlockchain(blockchainAddress);
        return externalBlockchain.sendData(msg.sender, _data);
    }

    // Function to receive data from an external blockchain
    function receiveDataFromExternalBlockchain(string memory _name, bytes calldata _data) public {
        address blockchainAddress = externalBlockchains[_name];
        require(blockchainAddress != address(0), "Blockchain not registered");

        IExternalBlockchain externalBlockchain = IExternalBlockchain(blockchainAddress);
        externalBlockchain.receiveData(msg.sender, _data);
    }
}
