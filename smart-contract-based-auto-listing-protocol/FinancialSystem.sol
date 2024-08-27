pragma solidity ^0.8.0;

contract FinancialSystem {
    // Mapping of financial system configurations
    mapping (address => mapping (address => bool)) public configurations;

    // Event emitted when a financial system configuration is updated
    event ConfigurationUpdated(address indexed piCoin, address indexed financialSystem);

    // Function to update a financial system configuration
    function updateConfiguration(address piCoin, address financialSystem) public {
        require(piCoin != address(0), "Pi Coin address cannot be zero");
        require(financialSystem != address(0), "Financial system address cannot be zero");
        configurations[piCoin][financialSystem] = true;
        emit ConfigurationUpdated(piCoin, financialSystem);
    }

    // Function to get the configuration status of a financial system
    function getConfigurationStatus(address piCoin, address financialSystem) public view returns (bool) {
        return configurations[piCoin][financialSystem];
    }
}
