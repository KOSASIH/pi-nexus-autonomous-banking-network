pragma solidity ^0.8.0;

import "./PiToken.sol";

contract PaymentGateway {
    // PiToken contract
    PiToken public piToken;

    // Mapping of merchant addresses to payment information
    mapping (address => PaymentInfo) public merchants;

    // Struct to store payment information
    struct PaymentInfo {
        uint256 paymentAmount;
        uint256 paymentCount;
        address[] paymentHistory;
    }

    // Event emitted when a payment is made
    event PaymentMade(address indexed merchant, address indexed customer, uint256 amount);

    // Event emitted when a merchant is registered
    event MerchantRegistered(address indexed merchant);

    // Constructor
    constructor() public {
        piToken = PiToken(address(new PiToken()));
    }

    // Register a merchant
    function registerMerchant(address merchant) public {
        require(merchants[merchant].paymentAmount == 0, "Merchant already registered");
        merchants[merchant] = PaymentInfo(0, 0, new address[](0));
        emit MerchantRegistered(merchant);
    }

    // Make a payment
    function makePayment(address merchant, uint256 amount) public {
        require(merchants[merchant].paymentAmount > 0, "Merchant not registered");
        require(piToken.balanceOf(msg.sender) >= amount, "Insufficient balance");
        piToken.transferFrom(msg.sender, merchant, amount);
        merchants[merchant].paymentAmount = merchants[merchant].paymentAmount.add(amount);
        merchants[merchant].paymentCount = merchants[merchant].paymentCount.add(1);
        merchants[merchant].paymentHistory.push(msg.sender);
        emit PaymentMade(merchant, msg.sender, amount);
    }

    // Get payment information for a merchant
    function getPaymentInfo(address merchant) public view returns (uint256, uint256, address[] memory) {
        return (merchants[merchant].paymentAmount, merchants[merchant].paymentCount, merchants[merchant].paymentHistory);
    }
}
