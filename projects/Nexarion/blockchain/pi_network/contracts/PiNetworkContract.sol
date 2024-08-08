pragma solidity ^0.8.0;

import "./PiTokenContract.sol";
import "./PiNetworkGovernanceContract.sol";

contract PiNetworkContract {
    // Mapping of user addresses to their PI balances
    mapping (address => uint256) public piBalances;

    // Mapping of user addresses to their node information
    mapping (address => NodeInfo) public nodeInfo;

    // PI token contract instance
    PiTokenContract public piTokenContract;

    // Governance contract instance
    PiNetworkGovernanceContract public governanceContract;

    // Event emitted when a new node is added to the network
    event NewNodeAdded(address indexed nodeAddress);

    // Event emitted when a node is removed from the network
    event NodeRemoved(address indexed nodeAddress);

    // Event emitted when a user's PI balance is updated
    event PiBalanceUpdated(address indexed userAddress, uint256 newBalance);

    // Constructor function
    constructor() public {
        // Initialize PI token contract instance
        piTokenContract = new PiTokenContract();

        // Initialize governance contract instance
        governanceContract = new PiNetworkGovernanceContract();
    }

    // Function to add a new node to the network
    function addNode(address nodeAddress) public {
        // Check if the node address is already in the network
        require(nodeInfo[nodeAddress].exists == false, "Node already exists in the network");

        // Create a new node information struct
        NodeInfo newNodeInfo = NodeInfo({
            exists: true,
            nodeAddress: nodeAddress,
            piBalance: 0
        });

        // Add the new node to the network
        nodeInfo[nodeAddress] = newNodeInfo;

        // Emit event to notify the network of the new node
        emit NewNodeAdded(nodeAddress);
    }

    // Function to remove a node from the network
    function removeNode(address nodeAddress) public {
        // Check if the node address exists in the network
        require(nodeInfo[nodeAddress].exists == true, "Node does not exist in the network");

        // Remove the node from the network
        delete nodeInfo[nodeAddress];

        // Emit event to notify the network of the node removal
        emit NodeRemoved(nodeAddress);
    }

    // Function to update a user's PI balance
    function updatePiBalance(address userAddress, uint256 newBalance) public {
        // Check if the user address exists in the network
        require(piBalances[userAddress] != 0, "User does not exist in the network");

        // Update the user's PI balance
        piBalances[userAddress] = newBalance;

        // Emit event to notify the network of the PI balance update
        emit PiBalanceUpdated(userAddress, newBalance);
    }
}

struct NodeInfo {
    bool exists;
    address nodeAddress;
    uint256 piBalance;
}
