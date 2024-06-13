pragma solidity ^0.8.0;

import "./TokenModel.sol";

contract TokenController {
    // Token model instance
    TokenModel public tokenModel;

    // Event emitted when tokens are minted
    event Mint(address user, uint256 amount);

    // Event emitted when tokens are burned
    event Burn(address user, uint256 amount);

    // Constructor
    constructor(address tokenModelAddress) public {
        tokenModel = TokenModel(tokenModelAddress);
    }

    // Function to mint tokens
    function mint(address user, uint256 amount) public {
        tokenModel.mint(user, amount);
        emit Mint(user, amount);
    }

    // Function to burn tokens
    function burn(address user, uint256 amount) public {
        tokenModel.burn(user, amount);
        emit Burn(user, amount);
    }

    // Function to get the balance of a user
    function getBalance(address user) public view returns (uint256) {
        return tokenModel.getBalance(user);
    }

    // Function to transfer tokens
    function transfer(address from, address to, uint256 amount) public {
        tokenModel.transfer(from, to, amount);
    }
}
