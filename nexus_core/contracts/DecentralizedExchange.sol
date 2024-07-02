pragma solidity ^0.8.0;

contract DecentralizedExchange {
    mapping (address => mapping (address => uint256)) public orders;

    constructor() {
        // Initialize order book
    }

    function placeOrder(address token, uint256 amount, uint256 price) public {
        orders[msg.sender][token] = amount;
    }

    function cancelOrder(address token) public {
        delete orders[msg.sender][token];
    }

    function executeTrade(address token, uint256 amount, uint256 price) public {
        // Execute trade logic
    }

    function getOrders(address account) public view returns (mapping (address => uint256)) {
        return orders[account];
    }
}
