pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedSupplyChainManagement {
    // Mapping of product IDs to product details
    mapping (uint256 => Product) public products;

    // Event emitted when a new product is added
    event NewProductAdded(uint256 productId, string name, string description, address manufacturer);

    // Function to add a new product
    function addNewProduct(string memory _name, string memory _description, address _manufacturer) public {
        // Create new product
        uint256 productId = products.length++;
        products[productId] = Product(_name, _description, _manufacturer, 0, 0, 0);

        // Emit new product added event
        emit NewProductAdded(productId, _name, _description, _manufacturer);
    }

    // Function to update a product's status
    function updateProductStatus(uint256 _productId, uint256 _status) public {
        // Check if product exists
        require(products[_productId].name != "", "Product does not exist");

        // Update product status
        products[_productId].status = _status;
    }

    // Function to get a product's details
    function getProductDetails(uint256 _productId) public view returns (string memory, string memory, address, uint256, uint256, uint256) {
        return (products[_productId].name, products[_productId].description, products[_productId].manufacturer, products[_productId].status, products[_productId].quantity, products[_productId].price);
    }

    // Struct to represent a product
    struct Product {
        string name;
        string description;
        address manufacturer;
        uint256 status;
        uint256 quantity;
        uint256 price;
    }
}
