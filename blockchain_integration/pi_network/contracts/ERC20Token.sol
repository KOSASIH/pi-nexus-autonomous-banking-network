// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Snapshot.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";

contract ERC20Token is ERC20, Ownable, Pausable, ERC20Burnable, ERC20Pausable, ERC20Snapshot, ERC20Votes {
    // Events
    event Minted(address indexed to, uint256 amount);
    event Burned(address indexed from, uint256 amount);

    constructor(uint256 initialSupply) ERC20("PiToken", "PIT") {
        _mint(msg.sender, initialSupply);
    }

    // Mint new tokens
    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
        emit Minted(to, amount);
    }

    // Override the _beforeTokenTransfer hook to implement pausable functionality
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal whenNotPaused override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }

    // Pause the contract
    function pause() public onlyOwner {
        _pause();
    }

    // Unpause the contract
    function unpause() public onlyOwner {
        _unpause();
    }

    // Snapshot the current state of the token
    function snapshot() public onlyOwner returns (uint256) {
        return _snapshot();
    }

    // Override required by Solidity for ERC20Votes
    function _afterTokenTransfer(address from, address to, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._afterTokenTransfer(from, to, amount);
    }

    // Burn tokens
    function burn(uint256 amount) public override {
        super.burn(amount);
        emit Burned(msg.sender, amount);
    }

    // Burn tokens from a specific address
    function burnFrom(address account, uint256 amount) public override {
        super.burnFrom(account, amount);
        emit Burned(account, amount);
    }
}
