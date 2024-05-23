pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract DEX {

    // The PI Network contract address
    address public piNetworkContract;

    // The fee for each trade
    uint public fee;

    // The mapping of orders
    struct Order {
        address trader;
        uint amount;
        uint price;
        bool filled;
    }
    mapping(address => Order[]) public orders;

    // The function to initialize the contract
    constructor(address _piNetworkContract, uint _fee) {
        piNetworkContract = _piNetworkContract;
        fee = _fee;
    }

    // The function to create a new order
    function createOrder(uint _amount, uint _price) external {
        IERC20 piToken = IERC20(piNetworkContract);
        require(piToken.balanceOf(msg.sender) >= _amount, "Insufficient PI token balance");

        Order memory order = Order({
            trader: msg.sender,
            amount: _amount,
            price: _price,
            filled: false
        });

        orders[msg.sender].push(order);

        piToken.transferFrom(msg.sender, address(this), _amount);
    }

    // The function to cancel an order
    function cancelOrder(uint _orderId) external {
        Order memory order = orders[msg.sender][_orderId];
        require(!order.filled, "Order already filled");

        IERC20 piToken = IERC20(piNetworkContract);
        piToken.transfer(msg.sender, order.amount);

        orders[msg.sender][_orderId] = Order({
            trader: msg.sender,
            amount: 0,
            price: 0,
            filled: true
        });
    }

    // The function to execute a trade
    function executeTrade(address _trader, uint _orderId) external {
        Order memory order = orders[_trader][_orderId];
        require(!order.filled, "Order already filled");

        IERC20 piToken = IERC20(piNetworkContract);
        require(piToken.balanceOf(msg.sender) >= order.amount, "Insufficient PI token balance");

        uint feeAmount = (order.amount * fee) / 100;
        uint totalAmount = order.amount + feeAmount;

        piToken.transferFrom(msg.sender, address(this), totalAmount);
        piToken.transfer(_trader, order.amount);

        orders[_trader][_orderId] = Order({
            trader: _trader,
            amount: 0,
            price: 0,
            filled: true
        });

        emit TradeExecuted(_trader, _orderId, order.amount, order.price);
    }

    // The event for when a trade is executed
    event TradeExecuted(address indexed trader, uint indexed orderId, uint amount, uint price);

}
