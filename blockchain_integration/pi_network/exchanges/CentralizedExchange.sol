// File: CentralizedExchange.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract CentralizedExchange is Ownable, ReentrancyGuard {
    // Mapping of users
    mapping (address => User) public users;

    // Mapping of orders
    mapping (uint256 => Order) public orders;

    // Event emitted when a new user is registered
    event NewUser(address indexed user);

    // Event emitted when a new order is placed
    event NewOrder(uint256 indexed orderId, address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is filled
    event FillOrder(uint256 indexed orderId, address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is cancelled
    event CancelOrder(uint256 indexed orderId, address indexed user);

    /**
     * @dev Represents a user on the centralized exchange
     */
    struct User {
        address wallet;
        string kycStatus;
        string amlStatus;
    }

    /**
     * @dev Represents an order on the centralized order book
     */
    struct Order {
        uint256 id;
        address user;
        uint256 amount;
        uint256 price;
        uint256 timestamp;
    }

    /**
     * @dev Registers a new user on the centralized exchange
     * @param _wallet The user's wallet address
     * @param _kycStatus The user's KYC status
     * @param _amlStatus The user's AML status
     */
    function registerUser(address _wallet, string memory _kycStatus, string memory _amlStatus) public {
        users[_wallet] = User(_wallet, _kycStatus, _amlStatus);
        emit NewUser(_wallet);
    }

    /**
     * @dev Places a new order on the centralized order book
     * @param _amount The amount of tokens to buy/sell
     * @param _price The price of the tokens
     */
    function placeOrder(uint256 _amount, uint256 _price) public {
        require(users[msg.sender].kycStatus == "approved" && users[msg.sender].amlStatus == "approved", "User not approved");
        uint256 orderId = orders.length++;
        orders[orderId] = Order(orderId, msg.sender, _amount, _price, block.timestamp);
        emit NewOrder(orderId, msg.sender, _amount, _price);
    }

    /**
     * @dev Fills an order on the centralized order book
     * @param _orderId The ID of the order to fill
     * @param _amount The amount of tokens to fill
     * @param _price The price of the tokens
     */
    function fillOrder(uint256 _orderId, uint256 _amount, uint256 _price) public {
        Order storage order = orders[_orderId];
        require(order.user == msg.sender && order.amount == _amount && order.price == _price, "Invalid order");
        delete orders[_orderId];
        emit FillOrder(_orderId, msg.sender, _amount, _price);
    }

    /**
     * @dev Cancels an order on the centralized order book
     * @param _orderId The ID of the order to cancel
     */
    function cancelOrder(uint256 _orderId) public {
        delete orders[_orderId];
        emit CancelOrder(_orderId, msg.sender);
    }
}
