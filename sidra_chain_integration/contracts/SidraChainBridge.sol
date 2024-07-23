pragma solidity ^0.8.0;

contract SidraChainBridge {
    address public autonomousBankingNetworkAddress;
    address public sidraChainAddress;

    constructor(address _autonomousBankingNetworkAddress, address _sidraChainAddress) public {
        autonomousBankingNetworkAddress = _autonomousBankingNetworkAddress;
        sidraChainAddress = _sidraChainAddress;
    }

    function transferFunds(address _recipient, uint256 _amount) public {
        // Call the Autonomous Banking Network's transferFunds function
        IAutonomousBankingNetwork(autonomousBankingNetworkAddress).transferFunds(_recipient, _amount);

        // Call the Sidra Chain's transferFunds function
        ISidraChain(sidraChainAddress).transferFunds(_recipient, _amount);
    }

    function getAccountBalance(address _accountAddress) public view returns (uint256) {
        // Query the Autonomous Banking Network's account balance
        uint256 abnBalance = IAutonomousBankingNetwork(autonomousBankingNetworkAddress).getAccountBalance(_accountAddress);

        // Query the Sidra Chain's account balance
        uint256 scBalance = ISidraChain(sidraChainAddress).getAccountBalance(_accountAddress);

        // Return the combined balance
        return abnBalance + scBalance;
    }
}
