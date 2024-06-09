pragma solidity ^0.8.0;

import "./PiNetwork.sol";
import "./PiNetworkRegistry.sol";
import "./PiNetworkBridge.sol";

contract PiNetworkAPI {
    // Mapping of Pi Network contracts to their APIs
    mapping (address => address) public piNetworkAPIs;

    // Event emitted when a Pi Network contract is API-enabled
    event PiNetworkAPIEnabledEvent(address indexed piNetwork, address indexed api);

    // Function to enable API for a Pi Network contract
    function enableAPI(PiNetwork piNetwork) public {
        piNetworkAPIs[address(piNetwork)] = msg.sender;
        emit PiNetworkAPIEnabledEvent(address(piNetwork), msg.sender);
    }

    // Function to get a Pi Network contract by API
    function getPiNetworkByAPI(address api) public view returns (PiNetwork) {
        return PiNetwork(piNetworkAPIs[api]);
    }

    // Function to call a Pi Network contract function through the API
    function callPiNetworkFunction(address piNetwork, bytes calldata data) public {
        PiNetwork(piNetwork).call(data);
    }
}
