// PiNexusAutonomousBankingNetwork.sol

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PiNexusAutonomousBankingNetwork {
    // Mapping of user addresses to their respective banking information
    mapping (address => BankingInfo) public bankingInfo;

    // Event emitted when a new user is registered
    event NewUserRegistered(address indexed user, string name, string email);

    // Event emitted when a transaction is processed
    event TransactionProcessed(address indexed from, address indexed to, uint256 amount);

    // Struct to represent banking information
    struct BankingInfo {
        string name;
        string email;
        uint256 balance;
        uint256[] transactionHistory;
    }

    // Function to register a new user
    function registerUser(string memory _name, string memory _email) public {
        // Validate input data
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(bytes(_email).length > 0, "Email cannot be empty");

        // Create a new banking info struct
        BankingInfo memory newBankingInfo = BankingInfo(_name, _email, 0, new uint256[](0));

        // Set the banking info for the user
        bankingInfo[msg.sender] = newBankingInfo;

        // Emit the NewUserRegistered event
        emit NewUserRegistered(msg.sender, _name, _email);
    }

    // Function to process a transaction
    function processTransaction(address _to, uint256 _amount) public {
        // Validate input data
        require(_to!= address(0), "Invalid recipient address");
        require(_amount > 0, "Invalid transaction amount");

        // Get the sender's banking info
        BankingInfo storage senderBankingInfo = bankingInfo[msg.sender];

        // Check if the sender has sufficient balance
        require(senderBankingInfo.balance >= _amount, "Insufficient balance");

        // Update the sender's balance
        senderBankingInfo.balance -= _amount;

        // Get the recipient's banking info
        BankingInfo storage recipientBankingInfo = bankingInfo[_to];

        // Update the recipient's balance
        recipientBankingInfo.balance += _amount;

        // Update the transaction history for both parties
        senderBankingInfo.transactionHistory.push(_amount);
        recipientBankingInfo.transactionHistory.push(_amount);

        // Emit the TransactionProcessed event
        emit TransactionProcessed(msg.sender, _to, _amount);
    }
}
