pragma solidity ^0.8.0;

import "./PIBank.sol";
import "./UserAuthentication.sol";

contract AccountManagement is UserAuthentication {
    // Function to update account information
    function updateAccountInfo(address user, string memory newName, string memory newEmail) public {
        require(msg.sender == user, "Unauthorized access");
        // Update account information in the PIBank contract
        PIBank(msg.sender).updateAccountInfo(newName, newEmail);
    }

    // Function to delete an account
    function deleteAccount(address user) public {
        require(msg.sender == user, "Unauthorized access");
        // Delete account in the PIBank contract
        PIBank(msg.sender).deleteAccount();
    }
}
