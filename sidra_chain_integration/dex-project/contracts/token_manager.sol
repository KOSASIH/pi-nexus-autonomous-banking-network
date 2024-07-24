pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract TokenManager is Ownable {
    // Mapping of token addresses to their respective token info
    mapping(address => TokenInfo) public tokenInfo;

    // Event emitted when a new token is added
    event NewToken(address indexed token, string name, string symbol, uint256 totalSupply);

    // Event emitted when a token is updated
    event UpdateToken(address indexed token, string name, string symbol, uint256 totalSupply);

    // Event emitted when a token is removed
    event RemoveToken(address indexed token);

    // Function to add a new token
    function addToken(address token, string memory name, string memory symbol, uint256 totalSupply) public onlyOwner {
        // Check if the token already exists
        require(tokenInfo[token] == TokenInfo(0), "Token already exists");

        // Create a new token info
        TokenInfo memory tokenInfo_ = TokenInfo(name, symbol, totalSupply);

        // Add the token info to the mapping
        tokenInfo[token] = tokenInfo_;

        // Emit the NewToken event
        emit NewToken(token, name, symbol, totalSupply);
    }

    // Function to update a token
    function updateToken(address token, string memory name, string memory symbol, uint256 totalSupply) public onlyOwner {
        // Check if the token exists
        require(tokenInfo[token] != TokenInfo(0), "Token does not exist");

        // Update the token info
        tokenInfo[token].name = name;
        tokenInfo[token].symbol = symbol;
        tokenInfo[token].totalSupply = totalSupply;

        // Emit the UpdateToken event
        emit UpdateToken(token, name, symbol, totalSupply);
    }

    // Function to remove a token
    function removeToken(address token) public onlyOwner {
        // Check if the token exists
        require(tokenInfo[token] != TokenInfo(0), "Token does not exist");

        // Remove the token info from the mapping
        delete tokenInfo[token];

        // Emit the RemoveToken event
        emit RemoveToken(token);
    }

    // Function to get a token info
    function getTokenInfo(address token) public view returns (TokenInfo memory) {
        return tokenInfo[token];
    }

    // TokenInfo struct
    struct TokenInfo {
        string name;
        string symbol;
        uint256 totalSupply;
    }
}
