pragma solidity ^0.8.0;

contract PiNetworkMarket {
    // Mapping of user addresses to their respective balances
    mapping (address => uint256) public balances;

    // Mapping of user addresses to their respective transaction histories
    mapping (address => Transaction[]) public transactionHistory;

    // Event emitted when a user deposits funds
    event Deposit(address indexed user, uint256 amount);

    // Event emitted when a user withdraws funds
    event Withdrawal(address indexed user, uint256 amount);

    // Event emitted when a trade is executed
    event Trade(address indexed buyer, address indexed seller, uint256 amount);

    // Struct to represent a transaction
    struct Transaction {
        uint256 amount;
        address sender;
        address recipient;
        uint256 timestamp;
    }

    // Function to deposit funds into the market
    function deposit() public payable {
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    // Function to withdraw funds from the market
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        emit Withdrawal(msg.sender, amount);
    }

    // Function to execute a trade between two users
    function trade(address buyer, address seller, uint256 amount) public {
        require(balances[buyer] >= amount, "Buyer has insufficient balance");
        require(balances[seller] >= amount, "Seller has insufficient balance");
        balances[buyer] -= amount;
        balances[seller] += amount;
        emit Trade(buyer, seller, amount);

        // Add transaction to buyer's and seller's transaction history
        transactionHistory[buyer].push(Transaction(amount, buyer, seller, block.timestamp));
        transactionHistory[seller].push(Transaction(amount, seller, buyer, block.timestamp));
    }

    // Function to get a user's transaction history
    function getTransactionHistory(address user) public view returns (Transaction[] memory) {
        return transactionHistory[user];
    }

    // Function to get a user's balance
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }
}
