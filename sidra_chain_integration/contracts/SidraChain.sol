pragma solidity ^0.8.0;

contract SidraChain {
    address public autonomousBankingNetworkAddress;

    constructor(address _autonomousBankingNetworkAddress) public {
        autonomousBankingNetworkAddress = _autonomousBankingNetworkAddress;
    }

    function transferFunds(address _recipient, uint256 _amount) public {
        // Call the autonomous banking network's transferFunds function
        IAutonomousBankingNetwork(autonomousBankingNetworkAddress).transferFunds(_recipient, _amount);
    }
}
