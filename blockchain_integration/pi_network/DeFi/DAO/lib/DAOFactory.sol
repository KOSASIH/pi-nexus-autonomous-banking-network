pragma solidity ^0.8.0;

library DAOFactory {
    // Function to create a new DAO
    function createDAO(string memory name, string memory symbol) public returns (address) {
        // Create a new DAO contract
        DAO dao = new DAO(name, symbol);

        // Return the DAO contract address
        return address(dao);
    }
}
