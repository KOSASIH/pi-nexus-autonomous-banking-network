pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiDEXV4 {
    using SafeMath for uint256;

    // Mapping of token listings
    mapping (address => TokenListing) public tokenListings;

    // Mapping of orders
    mapping (address => mapping (uint256 => Order)) public orders;

    // Event emitted when a new token is listed
    event TokenListed(address indexed token, uint256 listingId);

    // Event emitted when an order is placed
    event OrderPlaced(address indexed user, uint256 orderId, uint256 amount, uint256 price);

    // Event emitted when an order is matched
    event OrderMatched(uint256 orderId, uint256 amount, uint256 price);

    // Event emitted when a trade is executed
    event TradeExecuted(uint256 orderId, uint256 amount, uint256 price);

    // Struct to represent a token listing
    struct TokenListing {
        address token;
        uint256 listingId;
        uint256 decimals;
        uint256 liquidityProviderFee;
        uint256 makerFee;
        uint256 takerFee;
        uint256 maxOrderSize;
        uint256 minOrderSize;
    }

    // Struct to represent an order
    struct Order {
        uint256 orderId;
        address user;
        uint256 amount;
        uint256 price;
        uint256 timestamp;
        uint256 liquidityProviderFee;
        uint256 makerFee;
        uint256 takerFee;
        bool isMarketOrder;
    }

    // Advanced order matching algorithm (Hybrid of FIFO and Proportional Fairness)
    function matchOrders(address token, uint256 amount, uint256 price) internal {
        // Get the orders for the specified token
        Order[] memory ordersForToken = getOrdersForToken(token);

        // Initialize the proportional fairness variables
        uint256 totalWeight = 0;
        uint256[] memory weights = new uint256[](ordersForToken.length);

        // Calculate the weights for each order
        for (uint256 i = 0; i < ordersForToken.length; i++) {
            weights[i] = ordersForToken[i].amount.mul(ordersForToken[i].price);
            totalWeight = totalWeight.add(weights[i]);
        }

        // Calculate the proportional fairness score for each order
        uint256[] memory scores = new uint256[](ordersForToken.length);
        for (uint256 i = 0; i < ordersForToken.length; i++) {
            scores[i] = weights[i].mul(ordersForToken[i].timestamp) / totalWeight;
        }

        // Sort the orders by their scores
        ordersForToken = sortOrdersByScore(ordersForToken, scores);

        // Match the orders using the FIFO algorithm
        for (uint256 i = 0; i < ordersForToken.length; i++) {
            if (ordersForToken[i].amount <= amount) {
                // Match the entire order
                amount = amount.sub(ordersForToken[i].amount);
                emit OrderMatched(ordersForToken[i].orderId, ordersForToken[i].amount, ordersForToken[i].price);
                if (amount == 0) break;
            } else {
                // Match a portion of the order
                uint256 matchedAmount = amount;
                amount = 0;
                emit OrderMatched(ordersForToken[i].orderId, matchedAmount, ordersForToken[i].price);
                break;
            }
        }
    }

    // Function to list a new token
    function listToken(address token, uint256 decimals, uint256 liquidityProviderFee, uint256 makerFee, uint256 takerFee, uint256 maxOrderSize, uint256 minOrderSize) public {
        // Create a new token listing
        TokenListing memory listing = TokenListing(token, tokenListings[token].listingId + 1, decimals, liquidityProviderFee, makerFee, takerFee, maxOrderSize, minOrderSize);
        tokenListings[token] = listing;

        // Emit the TokenListed event
        emit TokenListed(token, listing.listingId);
    }

    // Function to place an order
    function placeOrder(address token, uint256 amount, uint256 price, uint256 liquidityProviderFee, uint256 makerFee, uint256 takerFee, bool isMarketOrder) public {
        // Create a new order
        Order memory order = Order(orders[token].length + 1, msg.sender, amount, price, block.timestamp, liquidityProviderFee, makerFee, takerFee, isMarketOrder);
        orders[token][order.orderId] = order;

        // Emit the OrderPlaced event
        emit OrderPlaced(msg.sender, order.orderId, amount, price);
    }

    // Function to execute a trade
    function executeTrade(address token, uint256 amount, uint256 price) public {
        // Check if the token is listed
        require(tokenListings[token].listingId > 0, "Token is not listed");

        // Check if the order is valid
        require(amount > 0 && price > 0, "Invalid order amount or price");

        // Check if the trader has enough tokens
        require(IERC20(token).balanceOf(msg.sender) >= amount, "Insufficient token balance");

        // Execute the trade using the advanced order matching algorithm
        matchOrders(token, amount, price);

        // Transfer the tokens from the trader to the liquidity provider
        IERC20(token).transferFrom(msg.sender, address(this), amount);
    }

    // Function to get the orders for a specified token
    function getOrdersForToken(address token) private view returns (Order[] memory) {
        Order[] memory ordersForToken = new Order[](orders[token].length);
        for (uint256 i = 0; i < orders[token].length; i++) {
            ordersForToken[i] = orders[token][i + 1];
        }
        return ordersForToken;
    }

    // Function to sort the orders by their scores
    function sortOrdersByScore(Order[] memory orders, uint256[] memory scores) private pure returns (Order[] memory) {
        // Create a copy of the orders array
        Order[] memory sortedOrders = new Order[](orders.length);
        for (uint256 i = 0; i < orders.length; i++) {
            sortedOrders[i] = orders[i];
        }

        // Sort the orders by their scores using the bubble sort algorithm
        for (uint256 i = 0; i < orders.length - 1; i++) {
            for (uint256 j = 0; j < orders.length - i - 1; j++) {
                if (scores[j] < scores[j + 1]) {
                    (sortedOrders[j], sortedOrders[j + 1]) = (sortedOrders[j + 1], sortedOrders[j]);
                    (scores[j], scores[j + 1]) = (scores[j + 1], scores[j]);
                }
            }
        }

        return sortedOrders;
    }

    // Function to get the token listing
    function getTokenListing(address token) public view returns (TokenListing memory) {
        return tokenListings[token];
    }

    // Function to get the order
    function getOrder(address token, uint256 orderId) public view returns (Order memory) {
        return orders[token][orderId];
    }

    // Function to cancel an order
    function cancelOrder(address token, uint256 orderId) public {
        // Check if the order exists
        require(orders[token][orderId].user == msg.sender, "Invalid order");

        // Remove the order from the orders mapping
        delete orders[token][orderId];
    }

    // Function to get the balance of a token
    function getTokenBalance(address token) public view returns (uint256) {
        return IERC20(token).balanceOf(address(this));
    }

    // Function to get the balance of a token for a user
    function getUserTokenBalance(address token, address user) public view returns (uint256) {
        return IERC20(token).balanceOf(user);
    }

    // Function to get the token decimals
    function getTokenDecimals(address token) public view returns (uint256) {
        return tokenListings[token].decimals;
    }

    // Function to get the liquidity provider fee
    function getLiquidityProviderFee(address token) public view returns (uint256) {
        return tokenListings[token].liquidityProviderFee;
    }

    // Function to get the maker fee
    function getMakerFee(address token) public view returns (uint256) {
        return tokenListings[token].makerFee;
    }

    // Function to get the taker fee
    function getTakerFee(address token) public view returns (uint256) {
        return tokenListings[token].takerFee;
    }

    // Function to get the max order size
    function getMaxOrderSize(address token) public view returns (uint256) {
        return tokenListings[token].maxOrderSize;
    }

    // Function to get the min order size
    function getMinOrderSize(address token) public view returns (uint256) {
        return tokenListings[token].minOrderSize;
    }
}
