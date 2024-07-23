pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNetworkUI {
    address private owner;
    PiNetwork private piNetwork;

    constructor() {
        owner = msg.sender;
    }

    function setPiNetwork(address _piNetwork) public {
        piNetwork = PiNetwork(_piNetwork);
    }

    function getDashboardData() public view returns (string memory) {
        // Return dashboard data in JSON format
        return piNetwork.getDashboardData();
    }

    function getWalletData() public view returns (string memory) {
        // Return wallet data in JSON format
        return piNetwork.getWalletData();
    }
}
