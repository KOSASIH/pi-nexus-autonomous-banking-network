// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CrossChainBridge {
    event TransferInitiated(address indexed sender, uint256 amount, string targetChain, address targetAddress);
    event TransferCompleted(address indexed receiver, uint256 amount, string sourceChain);

    struct Transfer {
        address sender;
        uint256 amount;
        string targetChain;
        address targetAddress;
        bool completed;
    }

    mapping(bytes32 => Transfer) public transfers;

    // Initiate a cross-chain transfer
    function initiateTransfer(uint256 _amount, string memory _targetChain, address _targetAddress) public {
        bytes32 transferId = keccak256(abi.encodePacked(msg.sender, _amount, _targetChain, _targetAddress, block.timestamp));
        require(transfers[transferId].sender == address(0), "Transfer already initiated.");

        transfers[transferId] = Transfer(msg.sender, _amount, _targetChain, _targetAddress, false);
        emit TransferInitiated(msg.sender, _amount, _targetChain, _targetAddress);
    }

    // Complete a cross-chain transfer
    function completeTransfer(bytes32 _transferId) public {
        Transfer storage transfer = transfers[_transferId];
        require(transfer.sender != address(0), "Transfer does not exist.");
        require(!transfer.completed, "Transfer already completed.");

        transfer.completed = true;
        payable(transfer.targetAddress).transfer(transfer.amount);
        emit TransferCompleted(transfer.targetAddress, transfer.amount, "Ethereum");
    }

    // Get transfer details
    function getTransferDetails(bytes32 _transferId) public view returns (address, uint256, string memory, address, bool) {
        Transfer storage transfer = transfers[_transferId];
        return (transfer.sender, transfer.amount, transfer.targetChain, transfer.targetAddress, transfer.completed);
    }

    // Fallback function to receive Ether
    receive() external payable {}
}
