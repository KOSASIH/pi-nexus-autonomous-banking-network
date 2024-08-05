pragma solidity ^0.8.0;

contract TokenContract {
    // Mapping of token holders to their token balance
    mapping (address => uint256) public tokenHolders;

    // Event emitted when tokens are transferred
    event TokensTransferred(address from, address to, uint256 amount);

    // Function to mint tokens
    function mintTokens(address recipient, uint256 amount) public {
        tokenHolders[recipient] += amount;
    }

    // Function to transfer tokens
    function transferTokens(address from, address to, uint256 amount) public {
        require(tokenHolders[from] >= amount, "From address does not have enough tokens");

        tokenHolders[from] -= amount;
        tokenHolders[to] += amount;

        emit TokensTransferred(from, to, amount);
    }
}
