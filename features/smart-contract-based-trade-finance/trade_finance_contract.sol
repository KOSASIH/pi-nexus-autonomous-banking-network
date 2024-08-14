pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract TradeFinanceContract is Ownable {
    using SafeMath for uint256;
    using Address for address;

    // Mapping of trade finance transactions
    mapping (address => mapping (address => TradeFinanceTransaction[])) public tradeFinanceTransactions;

    // Mapping of Allah features
    mapping (string => string) public allahFeatures;

    // Event emitted when a new trade finance transaction is created
    event NewTradeFinanceTransaction(address indexed buyer, address indexed seller, uint256 transactionId, uint256 amount);

    // Event emitted when a payment is made
    event PaymentMade(address indexed buyer, address indexed seller, uint256 transactionId, uint256 amount);

    // Event emitted when a dispute is raised
    event DisputeRaised(address indexed buyer, address indexed seller, uint256 transactionId);

    // Event emitted when a dispute is resolved
    event DisputeResolved(address indexed buyer, address indexed seller, uint256 transactionId);

    // Event emitted when an Allah feature is added
    event AllahFeatureAdded(string featureName, string featureDescription);

    // Struct to represent a trade finance transaction
    struct TradeFinanceTransaction {
        uint256 transactionId;
        uint256 amount;
        bool paymentMade;
        bool paymentReceived;
        bool disputeRaised;
        bool disputeResolved;
    }

    // Function to create a new trade finance transaction
    function createTradeFinanceTransaction(address buyer, address seller, uint256 amount) public onlyOwner {
        // Check if the buyer and seller are valid
        require(buyer != address(0) && seller != address(0), "Invalid buyer or seller");

        // Check if the amount is valid
        require(amount > 0, "Invalid amount");

        // Create a new transaction
        TradeFinanceTransaction memory transaction;
        transaction.transactionId = uint256(keccak256(abi.encodePacked(buyer, seller, amount)));
        transaction.amount = amount;
        transaction.paymentMade = false;
        transaction.paymentReceived = false;
        transaction.disputeRaised = false;
        transaction.disputeResolved = false;

        // Add the transaction to the mapping
        tradeFinanceTransactions[buyer][seller].push(transaction);

        // Emit an event to notify the creation of the transaction
        emit NewTradeFinanceTransaction(buyer, seller, transaction.transactionId, amount);
    }

    // Function to make a payment
    function makePayment(address buyer, address seller, uint256 transactionId, uint256 amount) public {
        // Check if the buyer and seller are valid
        require(buyer != address(0) && seller != address(0), "Invalid buyer or seller");

        // Check if the transaction ID is valid
        require(transactionId > 0, "Invalid transaction ID");

        // Check if the amount is valid
        require(amount > 0, "Invalid amount");

        // Get the transaction from the mapping
        TradeFinanceTransaction[] memory transactions = tradeFinanceTransactions[buyer][seller];

        // Find the transaction with the given ID
        for (uint256 i = 0; i < transactions.length; i++) {
            if (transactions[i].transactionId == transactionId) {
                // Check if the payment has already been made
                require(!transactions[i].paymentMade, "Payment already made");

                // Check if the payment amount matches the transaction amount
                require(amount == transactions[i].amount, "Invalid payment amount");

                // Make the payment
                transactions[i].paymentMade = true;

                // Emit an event to notify the payment
                emit PaymentMade(buyer, seller, transactionId, amount);

                // Return
                return;
            }
        }

        // If the transaction is not found, revert
        revert("Transaction not found");
    }

    // Function to receive a payment
    function receivePayment(address buyer, address seller, uint256 transactionId, uint256 amount) public {
        // Check if the buyer and seller are valid
        require(buyer != address(0) && seller != address(0), "Invalid buyer or seller");

        // Check if the transaction ID is valid
        require(transactionId > 0, "Invalid transaction ID");

        // Check if the amount is valid
        require(amount > 0, "Invalid amount");

        // Get the transaction from the mapping
        TradeFinanceTransaction[] memory transactions = tradeFinanceTransactions[buyer][seller];

        // Find the transaction with the given ID
        for (uint256 i = 0; i < transactions.length; i++) {
            if (transactions[i].transactionId == transactionId) {
                // Check if the payment has already been received
                require(!transactions[i].paymentReceived, "Payment already received");

                // Check if the payment amount matches the transaction amount
                require(amount == transactions[i].amount, "Invalid payment amount");

                // Receive the payment
                transactions[i].paymentReceived = true;

                // Emit an event to notify the payment receipt
                emit PaymentReceived(buyer, seller, transactionId, amount);

                // Return
                return;
            }
        }

        // If the transaction is not found, revert
        revert("Transaction not found");
    }

    // Function to raise a dispute
    function raiseDispute(address buyer, address seller, uint256 transactionId) public {
        // Check if the buyer and seller are valid
        require(buyer != address(0) && seller != address(0), "Invalid buyer or seller");

        // Check if the transaction ID is valid
        require(transactionId > 0, "Invalid transaction ID");

        // Get the transaction from the mapping
        TradeFinanceTransaction[] memory transactions = tradeFinanceTransactions[buyer][seller];

        // Find the transaction with the given ID
        for (uint256 i = 0; i < transactions.length; i++) {
            if (transactions[i].transactionId == transactionId) {
                // Check if the dispute has already been raised
                require(!transactions[i].disputeRaised, "Dispute already raised");

                // Raise the dispute
                transactions[i].disputeRaised = true;

                // Emit an event to notify the dispute
                emit DisputeRaised(buyer, seller, transactionId);

                // Return
                return;
            }
        }

        // If the transaction is not found, revert
        revert("Transaction not found");
    }

    // Function to resolve a dispute
    function resolveDispute(address buyer, address seller, uint256 transactionId) public {
        // Check if the buyer and seller are valid
        require(buyer != address(0) && seller != address(0), "Invalid buyer or seller");

        // Check if the transaction ID is valid
        require(transactionId > 0, "Invalid transaction ID");

        // Get the transaction from the mapping
        TradeFinanceTransaction[] memory transactions = tradeFinanceTransactions[buyer][seller];

        // Find the transaction with the given ID
        for (uint256 i = 0; i < transactions.length; i++) {
            if (transactions[i].transactionId == transactionId) {
                // Check if the dispute has already been resolved
                require(!transactions[i].disputeResolved, "Dispute already resolved");

                // Resolve the dispute
                transactions[i].disputeResolved = true;

                // Emit an event to notify the dispute resolution
                emit DisputeResolved(buyer, seller, transactionId);

                // Return
                return;
            }
        }

        // If the transaction is not found, revert
        revert("Transaction not found");
    }

    // Function to add an Allah feature
    function addAllahFeature(string memory featureName, string memory featureDescription) public onlyOwner {
        // Check if the feature name is valid
        require(bytes(featureName).length > 0, "Invalid feature name");

        // Check if the feature description is valid
        require(bytes(featureDescription).length > 0, "Invalid feature description");

        // Add the feature to the mapping
        allahFeatures[featureName] = featureDescription;

        // Emit an event to notify the addition of the feature
        emit AllahFeatureAdded(featureName, featureDescription);
    }

    // Function to get an Allah feature by name
    function getAllahFeatureByName(string memory featureName) public view returns (string memory) {
        // Get the feature from the mapping
        return allahFeatures[featureName];
    }

    // Function to get all Allah features
    function getAllAllahFeatures() public view returns (string[] memory) {
        // Get the features from the mapping
        string[] memory features;

        // Iterate over the mapping to get all features
        for (string memory featureName in allahFeatures) {
            features.push(featureName);
        }

        // Return the features
        return features;
    }

    // Function to get all trade finance transactions for a buyer
    function getTradeFinanceTransactionsByBuyer(address buyer) public view returns (TradeFinanceTransaction[] memory) {
        // Get the transactions from the mapping
        TradeFinanceTransaction[] memory transactions;

        // Iterate over the mapping to get all transactions for the buyer
        for (address seller in tradeFinanceTransactions[buyer]) {
            transactions = abi.encodePacked(transactions, tradeFinanceTransactions[buyer][seller]);
        }

        // Return the transactions
        return transactions;
    }

    // Function to get
