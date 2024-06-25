// File: HybridExchange.sol
pragma solidity ^0.8.0;

import "https://github.com/Chainlink/oracle-solidity/contracts/src/v0.8/ChainlinkClient.sol";

contract HybridExchange is ChainlinkClient {
    // Mapping of orders
    mapping (address => mapping (uint256 => Order)) public orders;

    // Mapping of oracle prices
    mapping (address => uint256) public oraclePrices;

    // Event emitted when a new order is placed
    event NewOrder(address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is filled
    event FillOrder(address indexed user, uint256 amount, uint256 price);

    // Event emitted when an order is cancelled
    event CancelOrder(address indexed user, uint256 amount);

    /**
     * @dev Represents an order on the hybrid order book
     */
    struct Order {
        uint256 amount;
        uint256 price;
        uint256 timestamp;
    }

    /**
     * @dev Places a new order on the hybrid order book
     * @param _amount The amount of tokens to buy/sell
     * @param _price The price of the tokens
     */
    function placeOrder(uint256 _amount, uint256 _price) public {
        orders[msg.sender][_amount] = Order(_amount, _price, block.timestamp);
        emit NewOrder(msg.sender, _amount, _price);
    }

    /**
     * @dev Fills an order on the hybrid order book
     * @param _user The user who placed the order
     * @param _amount The amount of tokens to fill
     * @param _price The price of the tokens
     */
    function fillOrder(address _user, uint256 _amount, uint256 _price) public {
        Order storage order = orders[_user][_amount];
        require(order.amount == _amount && order.price == _price, "Invalid order");
        delete orders[_user][_amount];
        emit FillOrder(_user, _amount, _price);
    }

    /**
     * @dev Cancels an order on the hybrid order book
     * @param _amount The amount of tokens to cancel
     */
    function cancelOrder(uint256 _amount) public {
        delete orders[msg.sender][_amount];
        emit CancelOrder(msg.sender, _amount);
    }

    /**
     * @dev Requests an oracle price update
     * @param _token The token to request a price for
     */
    function requestOraclePrice(address _token) public {
        Chainlink.Request memory req = buildChainlinkRequest(stringToBytes32("oracle"), this, "fulfillOraclePrice");
        req.add("token", _token);
        sendChainlinkRequestTo(oracle, req, "0.1 * 10 ** 18");
    }

    /**
     * @dev Fulfills an oracle priceupdate
     * @param _price The updated oracle price
     */
    function fulfillOraclePrice(bytes32 _requestId, uint256 _price) public recordChainlinkFulfillment(_requestId) {
        oraclePrices[msg.sender] = _price;
    }
}
