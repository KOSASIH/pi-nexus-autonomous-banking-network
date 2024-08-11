pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract DeFiBridge {
    // Mapping of DeFi protocols
    mapping (address => address) public defiProtocols;

    // Function to lend Pi-Stablecoin on a DeFi protocol
    function lend(address protocol, uint256 amount) public {
        // Approve protocol to spend Pi-Stablecoin
        ERC20(piStablecoin).approve(protocol, amount);
        // Call protocol's lending function
        DeFiProtocol(protocol).lend(amount);
    }
}
