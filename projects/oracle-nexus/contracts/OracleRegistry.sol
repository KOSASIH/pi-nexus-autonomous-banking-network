pragma solidity ^0.8.0;

contract OracleRegistry {
    // Mapping of oracles to their respective authorization status
    mapping (address => bool) public authorizedOracles;

    // Event emitted when an oracle is authorized
    event AuthorizedOracle(address indexed oracle);

    // Event emitted when an oracle is deauthorized
    event DeauthorizedOracle(address indexed oracle);

    // Constructor
    constructor() public {}

    // Function to authorize an oracle
    function authorizeOracle(address _oracle) public {
        // Only allow the contract owner to authorize oracles
        require(msg.sender == owner(), "Only the contract owner can authorize oracles");

        // Authorize the oracle
        authorizedOracles[_oracle] = true;

        // Emit the AuthorizedOracle event
        emit AuthorizedOracle(_oracle);
    }

    // Function to deauthorize an oracle
    function deauthorizeOracle(address _oracle) public {
        // Only allow the contract owner to deauthorize oracles
        require(msg.sender == owner(), "Only the contract owner can deauthorize oracles");

        // Deauthorize the oracle
        authorizedOracles[_oracle] = false;

        // Emit the DeauthorizedOracle event
        emit DeauthorizedOracle(_oracle);
    }

    // Function to check if an oracle is authorized
    function isAuthorizedOracle(address _oracle) public view returns (bool) {
        return authorizedOracles[_oracle];
    }
}
