// supply_chain_management.sol
pragma solidity ^0.6.0;

contract SupplyChainManagement {
  struct BankOperation {
    address bank;
    string operationType;
    uint256 timestamp;
  }

  mapping (address => BankOperation[]) public bankOperations;

  function addBankOperation(address bank, string memory operationType) public {
    BankOperation memory newOperation = BankOperation(bank, operationType, block.timestamp);
    bankOperations[bank].push(newOperation);
  }

  function getBankOperations(address bank) public view returns (BankOperation[] memory) {
    return bankOperations[bank];
  }
}
