pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract MultiChainSupport {
    // Mapping of supported chains
    mapping (address => Chain) public supportedChains;

    // Function to bridge assets between chains
    function bridgeAssets(address fromChain, address toChain, uint256 amount) public {
        // Get the bridge contract for the fromChain
        BridgeContract bridge = BridgeContract(supportedChains[fromChain].bridge);
        // Call the bridge contract to bridge assets
        bridge.bridgeAssets(toChain, amount);
    }
}
