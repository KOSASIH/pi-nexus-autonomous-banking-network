pragma solidity ^0.8.0;

import "./PiNetworkLogic.sol";
import "./PiNetworkOracle.sol";

contract PiNetworkRouter {
    address private owner;
    PiNetworkLogic private piNetworkLogic;
    PiNetworkOracle private piNetworkOracle;

    constructor() {
        owner = msg.sender;
    }

    function setPiNetworkLogic(address _piNetworkLogic) public {
        piNetworkLogic = PiNetworkLogic(_piNetworkLogic);
    }

    function setPiNetworkOracle(address _piNetworkOracle) public {
        piNetworkOracle = PiNetworkOracle(_piNetworkOracle);
    }

    function routeTransfer(address _from, address _to, uint256 _amount) public {
        // Route transfer logic
        uint256 price = piNetworkOracle.getPrice(_from);
        piNetworkLogic.transfer(_from, _to, _amount * price);
    }
}
