pragma solidity ^0.8.0;

contract FoodTracker {
    // Mapping of food items to their owners
    mapping (address => mapping (uint256 => FoodItem)) public foodItems;

    // Mapping of food item IDs to their metadata
    mapping (uint256 => FoodItemMetadata) public foodItemMetadata;

    // Event emitted when a new food item is created
    event FoodItemCreated(uint256 indexed foodItemId, address indexed owner);

    // Event emitted when a food item is updated
    event FoodItemUpdated(uint256 indexed foodItemId, address indexed owner);

    // Event emitted when a food item is deleted
    event FoodItemDeleted(uint256 indexed foodItemId, address indexed owner);

    // Struct to represent a food item
    struct FoodItem {
        uint256 id;
        string name;
        string description;
        uint256 quantity;
        uint256 price;
        address owner;
    }

    // Struct to represent food item metadata
    struct FoodItemMetadata {
        string origin;
        string ingredients;
        string allergens;
        string expirationDate;
    }

    // Function to create a new food item
    function createFoodItem(string memory _name, string memory _description, uint256 _quantity, uint256 _price) public {
        // Generate a unique ID for the food item
        uint256 foodItemId = uint256(keccak256(abi.encodePacked(_name, _description, _quantity, _price)));

        // Create a new food item
        FoodItem memory foodItem = FoodItem(foodItemId, _name, _description, _quantity, _price, msg.sender);

        // Add the food item to the mapping
        foodItems[msg.sender][foodItemId] = foodItem;

        // Add the food item metadata to the mapping
        foodItemMetadata[foodItemId] = FoodItemMetadata("", "", "", "");

        // Emit an event to notify that a new food item has been created
        emit FoodItemCreated(foodItemId, msg.sender);
    }

    // Function to update a food item
    function updateFoodItem(uint256 _foodItemId, string memory _name, string memory _description, uint256 _quantity, uint256 _price) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Update the food item
        foodItems[msg.sender][_foodItemId].name = _name;
        foodItems[msg.sender][_foodItemId].description = _description;
        foodItems[msg.sender][_foodItemId].quantity = _quantity;
        foodItems[msg.sender][_foodItemId].price = _price;

        // Emit an event to notify that a food item has been updated
        emit FoodItemUpdated(_foodItemId, msg.sender);
    }

    // Function to delete a food item
    function deleteFoodItem(uint256 _foodItemId) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Delete the food item
        delete foodItems[msg.sender][_foodItemId];

        // Emit an event to notify that a food item has been deleted
        emit FoodItemDeleted(_foodItemId, msg.sender);
    }

    // Function to get a food item
    function getFoodItem(uint256 _foodItemId) public view returns (FoodItem memory) {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Return the food item
        return foodItems[msg.sender][_foodItemId];
    }

    // Function to get food item metadata
    function getFoodItemMetadata(uint256 _foodItemId) public view returns (FoodItemMetadata memory) {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Return the food item metadata
        return foodItemMetadata[_foodItemId];
    }

    // Function to update food item metadata
    function updateFoodItemMetadata(uint256 _foodItemId, string memory _origin, string memory _ingredients, string memory _allergens, string memory _expirationDate) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Update the food item metadata
        foodItemMetadata[_foodItemId].origin = _origin;
        foodItemMetadata[_foodItemId].ingredients = _ingredients;
        foodItemMetadata[_foodItemId].allergens = _allergens;
        foodItemMetadata[_foodItemId].expirationDate = _expirationDate;
    }

        // Function to add a new food item batch
    function addFoodItemBatch(uint256 _foodItemId, uint256 _batchQuantity, string memory _batchDate) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Create a new food item batch
        FoodItemBatch memory foodItemBatch = FoodItemBatch(_batchQuantity, _batchDate);

        // Add the food item batch to the mapping
        foodItemBatches[msg.sender][_foodItemId].push(foodItemBatch);

        // Emit an event to notify that a new food item batch has been added
        emit FoodItemBatchAdded(_foodItemId, msg.sender);
    }

    // Function to get a food item batch
    function getFoodItemBatch(uint256 _foodItemId, uint256 _batchIndex) public view returns (FoodItemBatch memory) {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Return the food item batch
        return foodItemBatches[msg.sender][_foodItemId][_batchIndex];
    }

    // Function to update a food item batch
    function updateFoodItemBatch(uint256 _foodItemId, uint256 _batchIndex, uint256 _batchQuantity, string memory _batchDate) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Update the food item batch
        foodItemBatches[msg.sender][_foodItemId][_batchIndex].batchQuantity = _batchQuantity;
        foodItemBatches[msg.sender][_foodItemId][_batchIndex].batchDate = _batchDate;

        // Emit an event to notify that a food item batch has been updated
        emit FoodItemBatchUpdated(_foodItemId, msg.sender);
    }

    // Function to delete a food item batch
    function deleteFoodItemBatch(uint256 _foodItemId, uint256 _batchIndex) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Delete the food item batch
        delete foodItemBatches[msg.sender][_foodItemId][_batchIndex];

        // Emit an event to notify that a food item batch has been deleted
        emit FoodItemBatchDeleted(_foodItemId, msg.sender);
    }

    // Function to transfer a food item to another user
    function transferFoodItem(uint256 _foodItemId, address _to) public {
        // Check if the food item exists
        require(foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Transfer the food item to the new owner
        foodItems[_to][_foodItemId] = foodItems[msg.sender][_foodItemId];

        // Update the owner of the food item
        foodItems[msg.sender][_foodItemId].owner = _to;

        // Emit an event to notify that a food item has been transferred
        emit FoodItemTransferred(_foodItemId, msg.sender, _to);
    }
}

contract OrderManager {
    // Mapping of orders to their owners
    mapping (address => mapping (uint256 => Order)) public orders;

    // Mapping of order IDs to their metadata
    mapping (uint256 => OrderMetadata) public orderMetadata;

    // Event emitted when an order is created
    event OrderCreated(uint256 indexed orderId, address indexed owner);

    // Event emitted when an order is updated
    event OrderUpdated(uint256 indexed orderId, address indexed owner);

    // Event emitted when an order is deleted
    event OrderDeleted(uint256 indexed orderId, address indexed owner);

    // Struct to represent an order
    struct Order {
        uint256 id;
        string customerName;
        string orderDate;
        uint256 total;
        address owner;
    }

    // Struct to represent order metadata
    struct OrderMetadata {
        string shippingAddress;
        string paymentMethod;
    }

    // Function to create a new order
    function createOrder(string memory _customerName, string memory _orderDate, uint256 _total) public {
        // Generate a unique ID for the order
        uint256 orderId = uint256(keccak256(abi.encodePacked(_customerName, _orderDate, _total)));

        // Create a new order
        Order memory order = Order(orderId, _customerName, _orderDate, _total, msg.sender);

        // Add the order to the mapping
        orders[msg.sender][orderId] = order;

        // Add the order metadata to the mapping
        orderMetadata[orderId] = OrderMetadata("", "");

        // Emit an event to notify that a new order has been created
        emit OrderCreated(orderId, msg.sender);

          // Function to update an order
    function updateOrder(uint256 _orderId, string memory _customerName, string memory _orderDate, uint256 _total) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Update the order
        orders[msg.sender][_orderId].customerName = _customerName;
        orders[msg.sender][_orderId].orderDate = _orderDate;
        orders[msg.sender][_orderId].total = _total;

        // Emit an event to notify that an order has been updated
        emit OrderUpdated(_orderId, msg.sender);
    }

    // Function to delete an order
    function deleteOrder(uint256 _orderId) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Delete the order
        delete orders[msg.sender][_orderId];

        // Emit an event to notify that an order has been deleted
        emit OrderDeleted(_orderId, msg.sender);
    }

    // Function to get an order
    function getOrder(uint256 _orderId) public view returns (Order memory) {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Return the order
        return orders[msg.sender][_orderId];
    }

    // Function to get order metadata
    function getOrderMetadata(uint256 _orderId) public view returns (OrderMetadata memory) {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Return the order metadata
        return orderMetadata[_orderId];
    }

    // Function to update order metadata
    function updateOrderMetadata(uint256 _orderId, string memory _shippingAddress, string memory _paymentMethod) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Update the order metadata
        orderMetadata[_orderId].shippingAddress = _shippingAddress;
        orderMetadata[_orderId].paymentMethod = _paymentMethod;
    }

    // Function to add a new order item
    function addOrderItem(uint256 _orderId, uint256 _foodItemId, uint256 _quantity) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Check if the food item exists
        require(FoodTracker(msg.sender).foodItems[msg.sender][_foodItemId].id != 0, "Food item does not exist");

        // Create a new order item
        OrderItem memory orderItem = OrderItem(_foodItemId, _quantity);

        // Add the order item to the mapping
        orderItems[msg.sender][_orderId].push(orderItem);

        // Emit an event to notify that a new order item has been added
        emit OrderItemAdded(_orderId, msg.sender);
    }

    // Function to get an order item
    function getOrderItem(uint256 _orderId, uint256 _orderItemIndex) public view returns (OrderItem memory) {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Return the order item
        return orderItems[msg.sender][_orderId][_orderItemIndex];
    }

    // Function to update an order item
    function updateOrderItem(uint256 _orderId, uint256 _orderItemIndex, uint256 _quantity) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Update the order item
        orderItems[msg.sender][_orderId][_orderItemIndex].quantity = _quantity;

        // Emit an event to notify that an order item has been updated
        emit OrderItemUpdated(_orderId, msg.sender);
    }

    // Function to delete an order item
    function deleteOrderItem(uint256 _orderId, uint256 _orderItemIndex) public {
        // Check if the order exists
        require(orders[msg.sender][_orderId].id != 0, "Order does not exist");

        // Delete the order item
        delete orderItems[msg.sender][_orderId][_orderItemIndex];

        // Emit an event to notify that an order item has been deleted
        emit OrderItemDeleted(_orderId, msg.sender);
    }
}
