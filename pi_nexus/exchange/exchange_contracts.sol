pragma solidity ^0.8.0;

interface Token {
    function transferFrom(address from, address to, uint256 value) external returns (bool success);
}

contract ExchangeContract {
    struct Order {
        address user;
        uint256 amount;
        uint256 price;
        uint256 timestamp;
        bool filled;
    }

    mapping(address => mapping(uint256 => Order)) public orders;

    event OrderCreated(address indexed user, uint256 indexed orderId, uint256 amount, uint256 price, uint256 timestamp);

    function createOrder(address _token, uint256 _amount, uint256 _price) public {
        Order memory newOrder;
        newOrder.user = msg.sender;
        newOrder.amount = _amount;
        newOrder.price = _price;
        newOrder.timestamp = block.timestamp;
        newOrder.filled = false;

        uint256 orderId = orders[msg.sender].length;
        orders[msg.sender][orderId] = newOrder;

        emit OrderCreated(msg.sender, orderId, _amount, _price, block.timestamp);
    }

    function fillOrder(address _token, uint256 _orderId, uint256 _amount) public {
        Order storage order = orders[msg.sender][_orderId];
        require(!order.filled, "This order has already been filled.");

        Token(_token).transferFrom(msg.sender, address(this), _amount);

        order.filled = true;
    }

    function getOrder(address _user, uint256 _orderId) public view returns (address, uint256, uint256, uint256, bool) {
        Order storage order = orders[_user][_orderId];
        return (order.user, order.amount, order.price, order.timestamp, order.filled);
    }
}
