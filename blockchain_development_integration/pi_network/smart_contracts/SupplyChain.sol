// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SupplyChain {
    struct Product {
        string name;
        address currentOwner;
        string status;
    }

    mapping(uint256 => Product) public products;
    uint256 public productCount;

    event ProductCreated(uint256 indexed productId, string name, address indexed owner);
    event ProductTransferred(uint256 indexed productId, address indexed from, address indexed to);

    function createProduct(string memory name) external {
        productCount++;
        products[productCount] = Product(name, msg.sender, "Created");
        emit ProductCreated(productCount, name, msg.sender);
    }

    function transferProduct(uint256 productId, address newOwner) external {
        require(products[productId].currentOwner == msg.sender, "Not the owner");
        products[productId].currentOwner = newOwner;
        products[productId].status = "Transferred";
        emit ProductTransferred(productId, msg.sender, newOwner);
    }

    function getProduct(uint256 productId) external view returns (string memory, address, string memory) {
        Product memory product = products[productId];
        return (product.name, product.currentOwner, product.status);
    }
}
