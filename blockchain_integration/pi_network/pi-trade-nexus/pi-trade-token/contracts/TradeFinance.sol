pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "./PiTradeToken.sol";

contract TradeFinance is Ownable {
    // Mapping of trade finance contracts
    mapping (address => TradeFinanceContract) public contracts;

    // Event emitted when a new trade finance contract is created
    event NewContract(address indexed creator, address indexed contractAddress);

    /**
     * @dev Creates a new trade finance contract
     * @param creator The address that created the contract
     * @param contractAddress The address of the new contract
     */
        function getContract(address creator) public view returns (TradeFinanceContract) {
        return contracts[creator];
    }
}

contract TradeFinanceContract {
    // Mapping of trade finance data
    mapping (address => TradeFinanceData) public data;

    // Event emitted when trade finance data is updated
    event UpdateData(address indexed creator, TradeFinanceData data);

    /**
     * @dev Initializes the contract with the creator's address
     * @param creator The address that created the contract
     */
    constructor(address creator) public {
        data[creator] = TradeFinanceData(0, 0, 0);
    }

    /**
     * @dev Updates the trade finance data
     * @param creator The address that created the contract
     * @param data The new trade finance data
     */
    function updateData(address creator, TradeFinanceData data) public {
        require(msg.sender == creator, "Only the creator can update the data");
        data[creator] = data;
        emit UpdateData(creator, data);
    }

    /**
     * @dev Gets the trade finance data
     * @param creator The address that created the contract
     * @return The trade finance data
     */
    function getData(address creator) public view returns (TradeFinanceData) {
        return data[creator];
    }
}

struct TradeFinanceData {
    uint256 totalValue;
    uint256 totalQuantity;
    uint256 averagePrice;
}
   
