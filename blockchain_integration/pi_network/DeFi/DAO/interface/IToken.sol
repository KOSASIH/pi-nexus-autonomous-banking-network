pragma solidity ^0.8.0;

interface IToken {
    // Function to mint tokens
    function mint(address user, uint256 amount) external;

    // Function to burn tokens
    function burn(address user, uint256 amount) external;

    // Function to get the balance of a user
    function getBalance(address user) external view returns (uint256);

    // Function to transfer tokens
    function transfer(address from, address to, uint256 amount) external;

    // Event emitted when tokens are minted
    event Mint(address user, uint256 amount);

    // Event emitted when tokens are burned
    event Burn(address user, uint256 amount);
}
