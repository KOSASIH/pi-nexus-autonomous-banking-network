pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract OrderBook is Ownable, ReentrancyGuard {
    // Mapping of asset pairs to their respective order books
    mapping(address => mapping(address => Order[])) public orderBooks;

    // Mapping of order IDs to their respective orders
    mapping(uint256 => Order) public orders;

    // Next order ID
    uint256 public nextOrderId;

    // Event emitted when a new order is placed
    event NewOrder(uint256 orderId, address indexed asset, address indexed maker, uint256 amount, uint256 price);

    // Event emitted when an order is cancelled
    event CancelOrder(uint256 orderId, address indexed asset, address indexed maker, uint256 amount);

    // Event emitted when an order is filled
    event FillOrder(uint256 orderId, address indexed asset, address indexed maker, address indexed taker, uint256 amount, uint256 price);

    // Function to place a new order
    function placeOrder(address asset, uint256 amount, uint256 price) public {
        // Check if the asset is supported
        require(asset != address(0), "Asset not supported");

        // Create a new order
        Order memory order = Order(nextOrderId, asset, msg.sender, amount, price);

        // Add the order to the order book
        orderBooks[asset][msg.sender].push(order);

        // Add the order to the orders mapping
        orders[nextOrderId] = order;

        // Increment the next order ID
        nextOrderId++;

        // Emit the NewOrder event
        emit NewOrder(nextOrderId - 1, asset, msg.sender, amount, price);
    }

    // Function to cancel an order
    function cancelOrder(uint256 orderId) public {
        // Check if the order exists
        require(orders[orderId] != Order(0), "Order does not exist");

        // Get the order
        Order memory order = orders[orderId];

        // Remove the order from the order book
        for (uint256 i = 0; i < orderBooks[order.asset][order.maker].length; i++) {
            if (orderBooks[order.asset][order.maker][i].id == orderId) {
                orderBooks[order.asset][order.maker][i] = orderBooks[order.asset][order.maker][orderBooks[order.asset][order.maker].length - 1];
                orderBooks[order.asset][order.maker].pop();
                break;
            }
        }

        // Remove the order from the orders mapping
        delete orders[orderId];

        // Emit the CancelOrder event
        emit CancelOrder(orderId, order.asset, order.maker, order.amount);
    }

    // Function to fill an order
    function fillOrder(uint256 orderId, uint256 amount) public {
        // Check if the order exists
        require(orders[orderId] != Order(0), "Order does not exist");

        // Get the order
        Order memory order = orders[orderId];

        // Check if the order can be filled
        require(order.amount >= amount, "Insufficient amount");

        // Update the order amount
        order.amount -= amount;

        // If the order is fully filled, remove it from the order book
        if (order.amount == 0) {
            cancelOrder(orderId);
        }

        // Emit the FillOrder event
        emit FillOrder(orderId, order.asset, order.maker, msg.sender, amount, order.price);
    }

    // Order struct
    struct Order {
        uint256 id;
        address asset;
        address maker;
        uint256 amount;
        uint256 price;
    }
}
