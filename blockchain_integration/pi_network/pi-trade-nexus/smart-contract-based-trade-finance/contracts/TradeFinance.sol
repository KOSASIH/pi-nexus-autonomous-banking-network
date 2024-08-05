pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "./PiTradeToken.sol";

contract TradeFinance {
    // Mapping of trade finance agreements
    mapping (address => TradeFinanceAgreement) public tradeFinanceAgreements;

    // Event emitted when a new trade finance agreement is created
    event NewTradeFinanceAgreement(address indexed buyer, address indexed seller, uint256 amount);

    // Event emitted when a trade finance agreement is fulfilled
    event TradeFinanceAgreementFulfilled(address indexed buyer, address indexed seller, uint256 amount);

    // Struct to represent a trade finance agreement
    struct TradeFinanceAgreement {
        address buyer;
        address seller;
        uint256 amount;
        uint256 expirationDate;
        bool fulfilled;
    }

    // Function to create a new trade finance agreement
    function createTradeFinanceAgreement(address buyer, address seller, uint256 amount, uint256 expirationDate) public {
        require(buyer!= address(0), "Buyer address cannot be zero");
        require(seller!= address(0), "Seller address cannot be zero");
        require(amount > 0, "Amount must be greater than zero");

        TradeFinanceAgreement storage agreement = tradeFinanceAgreements[buyer];
        agreement.buyer = buyer;
        agreement.seller = seller;
        agreement.amount = amount;
        agreement.expirationDate = expirationDate;
        agreement.fulfilled = false;

        emit NewTradeFinanceAgreement(buyer, seller, amount);
    }

    // Function to fulfill a trade finance agreement
    function fulfillTradeFinanceAgreement(address buyer, address seller, uint256 amount) public {
        require(buyer!= address(0), "Buyer address cannot be zero");
        require(seller!= address(0), "Seller address cannot be zero");
        require(amount > 0, "Amount must be greater than zero");

        TradeFinanceAgreement storage agreement = tradeFinanceAgreements[buyer];
        require(agreement.buyer == buyer, "Buyer does not have an active trade finance agreement");
        require(agreement.seller == seller, "Seller does not match the trade finance agreement");
        require(agreement.amount == amount, "Amount does not match the trade finance agreement");
        require(!agreement.fulfilled, "Trade finance agreement has already been fulfilled");

        // Transfer PiTradeToken from buyer to seller
        PiTradeToken.transferFrom(buyer, seller, amount);

        agreement.fulfilled = true;

        emit TradeFinanceAgreementFulfilled(buyer, seller, amount);
    }
}
