pragma solidity ^0.8.0;

import "./PiNetworkData.sol";

contract PiNetworkLogic {
    address private owner;
    PiNetworkData private piNetworkData;

    constructor() {
        owner = msg.sender;
    }

    function setPiNetworkData(address _piNetworkData) public {
        piNetworkData = PiNetworkData(_piNetworkData);
    }

    function transfer(address _from, address _to, uint256 _amount) public {
        // Transfer logic
        piNetworkData.setBalance(_from, piNetworkData.getBalance(_from) - _amount);
        piNetworkData.setBalance(_to, piNetworkData.getBalance(_to) + _amount);
    }
}
