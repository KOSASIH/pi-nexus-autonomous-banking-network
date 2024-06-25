// File: OrderBook.sol
pragma solidity ^0.8.0;

import "https://github.com/Uniswap/uniswap-v2-core/contracts/libraries/Math.sol";
import "https://github.com/Uniswap/uniswap-v2-core/contracts/libraries/UQ112x112.sol";

contract OrderBook {
    using Math for uint256;
    using UQ112x112 for uint224;

    // Mapping of orders
    mapping (address => mapping (uint256 => Order)) public orders;

    // Event emitted when a new order is placed
    event NewOrder(address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is filled
    event FillOrder(address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is cancelled
    event CancelOrder(address indexed user, uint256 amount);

    /**
     * @dev Represents an order on the order book
     */
    struct Order {
        uint256 amount;
        uint256 price;
        uint256 timestamp;
    }

    /**
     * @dev Places a new order on the order book
     * @param _amount The amount of tokens to buy/sell
     * @param _price The price of the tokens
     */
    function placeOrder(uint256 _amount, uint256 _price) public {
        orders[msg.sender][_amount] = Order(_amount, _price, block.timestamp);
        emit NewOrder(msg.sender, _amount, _price);
    }

   
