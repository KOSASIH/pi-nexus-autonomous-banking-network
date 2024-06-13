pragma solidity ^0.8.0;

contract TokenModel {
    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Event emitted when tokens are minted
    event Mint(address user, uint256 amount);

    // Event emittedwhen tokens are burned
    event Burn(address user, uint256 amount);

    // Function to mint tokens
    function mint(address user, uint256 amount) public {
        balances[user] += amount;
        emit Mint(user, amount);
    }

    // Function to burn tokens
    function burn(address user, uint256 amount) public {
        require(balances[user] >= amount, "Insufficient balance");
        balances[user] -= amount;
        emit Burn(user, amount);
    }

    // Function to get the balance of a user
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // Function to transfer tokens
    function transfer(address from, address to, uint256 amount) public {
        require(balances[from] >= amount, "Insufficient balance");
        balances[from] -= amount;
        balances[to] += amount;
    }
}
