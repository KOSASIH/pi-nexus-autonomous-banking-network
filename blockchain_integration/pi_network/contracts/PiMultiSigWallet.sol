pragma solidity ^0.8.0;

import "./PiToken.sol";

contract PiMultiSigWallet {
    address[] public owners;
    uint256 public requiredSignatures;
    mapping (address => bool) public isOwner;
    mapping (uint256 => Transaction) public transactions;

    struct Transaction {
        address from;
        address to;
        uint256 value;
        uint256 signatures;
    }

    constructor(address[] memory _owners, uint256 _requiredSignatures) public {
        owners = _owners;
        requiredSignatures = _requiredSignatures;
        for (address owner : owners) {
            isOwner[owner] = true;
        }
    }

    function submitTransaction(address _to, uint256 _value) public {
        require(isOwner[msg.sender], "Only owners can submit transactions");
        uint256 transactionId = transactions.length++;
        transactions[transactionId] = Transaction(msg.sender, _to, _value, 0);
    }

    function signTransaction(uint256 _transactionId) public {
        require(isOwner[msg.sender], "Only owners can sign transactions");
        Transaction storage transaction = transactions[_transactionId];
        require(transaction.signatures < requiredSignatures, "Transaction already signed");
        transaction.signatures++;
        if (transaction.signatures == requiredSignatures) {
            // Execute the transaction
            piToken.transfer(transaction.to, transaction.value);
        }
    }
}
