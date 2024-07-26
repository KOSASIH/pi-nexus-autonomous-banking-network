pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Bridge {
    // Mapping of token addresses to their respective bridge contracts
    mapping (address => address) public tokenToBridge;

    // Event emitted when a token is bridged
    event TokenBridged(address indexed token, address indexed recipient, uint256 amount);

    // Function to bridge a token
    function bridgeToken(address token, address recipient, uint256 amount) public {
        // Check if the token is supported
        require(tokenToBridge[token] != address(0), "Unsupported token");

        // Transfer the token to the bridge contract
        SafeERC20.safeTransfer(token, tokenToBridge[token], amount);

        // Emit the TokenBridged event
        emit TokenBridged(token, recipient, amount);
    }
}
