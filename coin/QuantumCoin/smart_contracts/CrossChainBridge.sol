// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract CrossChainBridge is Ownable {
    // Event emitted when assets are locked
    event AssetsLocked(address indexed sender, uint256 amount, string destinationChain);
    
    // Event emitted when assets are minted on the destination chain
    event AssetsMinted(address indexed recipient, uint256 amount, string sourceChain);

    // Mapping to track locked assets
    mapping(address => uint256) public lockedAssets;

    // Lock assets on the source chain
    function lockAssets(address token, uint256 amount, string memory destinationChain) external {
        require(amount > 0, "Amount must be greater than zero");
        
        // Transfer tokens to this contract
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        
        // Update locked assets
        lockedAssets[token] += amount;

        emit AssetsLocked(msg.sender, amount, destinationChain);
    }

    // Mint equivalent assets on the destination chain
    function mintAssets(address recipient, uint256 amount, string memory sourceChain) external onlyOwner {
        require(amount > 0, "Amount must be greater than zero");
        
        // Mint new tokens to the recipient
        // This function should be implemented in the token contract
        // Example: IERC20(token).mint(recipient, amount);
        
        emit AssetsMinted(recipient, amount, sourceChain);
    }

    // Withdraw locked assets (only for owner)
    function withdrawLockedAssets(address token, uint256 amount) external onlyOwner {
        require(lockedAssets[token] >= amount, "Insufficient locked assets");
        
        lockedAssets[token] -= amount;
        IERC20(token).transfer(msg.sender, amount);
    }
}
