// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract OrderBookDEX is Ownable {
    using SafeERC20 for IERC20;

    struct Order {
        address user;
        IERC20 tokenIn;
        IERC20 tokenOut;
        uint256 amountIn;
        uint256 amountOut;
        uint256 price;
        uint256 timestamp;
        bool filled;
    }

    mapping(address => mapping(address => Order[])) public orders;

    event OrderCreated(address indexed user, address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 amountOut, uint256 price, uint256 timestamp);
    event OrderFilled(address indexed user, address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 amountOut, uint256 price, uint256 timestamp);

    function createOrder(IERC20 tokenIn, IERC20 tokenOut, uint256 amountIn, uint256 amountOut, uint256 price) external {
        require(tokenIn != tokenOut, "Tokens must be different");
        require(amountIn > 0 && amountOut > 0, "Amounts must be greater than zero");
        require(price > 0, "Price must be greater than zero");

        Order memory newOrder;
        newOrder.user = msg.sender;
        newOrder.tokenIn = tokenIn;
        newOrder.tokenOut = tokenOut;
        newOrder.amountIn = amountIn;
        newOrder.amountOut = amountOut;
        newOrder.price = price;
        newOrder.timestamp = block.timestamp;
        newOrder.filled = false;

        orders[address(tokenIn)][address(tokenOut)].push(newOrder);

        emit OrderCreated(msg.sender, address(tokenIn), address(tokenOut), amountIn, amountOut, price, block.timestamp);
    }

    function fillOrder(address tokenIn, address tokenOut, uint256 index) external {
        Order storage order = orders[address(tokenIn)][address(tokenOut)][index];
        require(order.filled == false, "Order already filled");

        require(order.tokenIn.transferFrom(order.user, address(this), order.amountIn), "Transfer failed");
        require(order.tokenOut.transfer(order.user, order.amountOut), "Transfer failed");

        order.filled = true;

        emit OrderFilled(order.user, address(order.tokenIn), address(order.tokenOut), order.amountIn, order.amountOut, order.price, block.timestamp);
    }

    function cancelOrder(address tokenIn, address tokenOut, uint256 index) external {
        Order storage order = orders[address(tokenIn)][address(tokenOut)][index];
        require(order.filled == false, "Order already filled");

        order.tokenIn.safeTransfer(order.user, order.amountIn);
        order.tokenOut.safeTransfer(order.user, order.amountOut);

        delete orders[address(tokenIn)][address(tokenOut)][index];
    }
}
