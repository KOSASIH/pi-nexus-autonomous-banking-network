pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract KosasihUniversalisNexus {
    address public smartContractRegistry;
    address public tokenManager;

    constructor(address _smartContractRegistry, address _tokenManager) public {
        smartContractRegistry = _smartContractRegistry;
        tokenManager = _tokenManager;
    }

    function executeTransaction(bytes _transactionData) public {
        // Decode the transaction data
        (address _contractAddress, bytes _contractData) = abi.decode(_transactionData, (address, bytes));

        // Call the smart contract registry to get the contract ABI
        SmartContractRegistry(smartContractRegistry).getContractABI(_contractAddress);

        // Call the contract with the decoded data
        (bool success, ) = _contractAddress.call(_contractData);
        require(success, "Transaction execution failed");
    }
}
