// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "./Token.sol";

contract TokenFactory {
    // Mapping of created tokens
    mapping (address => Token) public tokens;

    // Event
    event NewToken(address indexed tokenAddress, string name, string symbol, uint8 decimals, uint256 totalSupply);

    // Create a new token
    function createToken(string memory_name, string memory _symbol, uint8 _decimals, uint256 _totalSupply) public {
        require(msg.sender == owner, "Only the owner can create tokens");
        Token token = new Token(_name, _symbol, _decimals, _totalSupply);
        tokens[address(token)] = token;
        emit NewToken(address(token), _name, _symbol, _decimals, _totalSupply);
    }

    // Get the token at a given address
    function getToken(address _tokenAddress) public view returns (Token) {
        return tokens[_tokenAddress];
    }
}
