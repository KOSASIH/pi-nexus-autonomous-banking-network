pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/Neural/Neural.sol";

contract PiNetworkNeural is Neural {
    // Mapping of user addresses to their neural networks
    mapping (address => NeuralNetwork) public neuralNetworks;

    // Struct to represent a neural network
    struct NeuralNetwork {
        string networkType;
        string networkData;
    }

    // Event emitted when a new neural network is created
    event NeuralNetworkCreatedEvent(address indexed user, NeuralNetwork network);

    // Function to create a new neural network
    function createNeuralNetwork(string memory networkType, string memory networkData) public {
        NeuralNetwork storage network = neuralNetworks[msg.sender];
        network.networkType = networkType;
        network.networkData = networkData;
        emit NeuralNetworkCreatedEvent(msg.sender, network);
    }

    // Function to get a neural network
    function getNeuralNetwork(address user) public view returns (NeuralNetwork memory) {
        return neuralNetworks[user];
    }
}
