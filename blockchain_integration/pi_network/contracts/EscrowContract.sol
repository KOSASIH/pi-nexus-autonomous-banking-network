// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract EscrowContract is Ownable {
    enum EscrowStatus { Created, Funded, Completed, Disputed, Refunded }

    struct Escrow {
        address buyer;
        address seller;
        address arbiter;
        uint256 amount;
        EscrowStatus status;
    }

    mapping(uint256 => Escrow) public escrows;
    uint256 public escrowCount;

    event EscrowCreated(uint256 indexed escrowId, address indexed buyer, address indexed seller, uint256 amount);
    event EscrowFunded(uint256 indexed escrowId);
    event EscrowCompleted(uint256 indexed escrowId);
    event EscrowDisputed(uint256 indexed escrowId);
    event EscrowRefunded(uint256 indexed escrowId);

    modifier onlyBuyer(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].buyer, "Only the buyer can call this function.");
        _;
    }

    modifier onlySeller(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].seller, "Only the seller can call this function.");
        _;
    }

    modifier onlyArbiter(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].arbiter, "Only the arbiter can call this function.");
        _;
    }

    modifier inStatus(uint256 escrowId, EscrowStatus status) {
        require(escrows[escrowId].status == status, "Invalid escrow status.");
        _;
    }

    function createEscrow(address seller, address arbiter) public payable returns (uint256) {
        require(msg.value > 0, "Escrow amount must be greater than zero.");
        escrowCount++;
        escrows[escrowCount] = Escrow({
            buyer: msg.sender,
            seller: seller,
            arbiter: arbiter,
            amount: msg.value,
            status: EscrowStatus.Created
        });

        emit EscrowCreated(escrowCount, msg.sender, seller, msg.value);
        return escrowCount;
    }

    function fundEscrow(uint256 escrowId) public onlyBuyer(escrowId) inStatus(escrowId, EscrowStatus.Created) {
        escrows[escrowId].status = EscrowStatus.Funded;
        emit EscrowFunded(escrowId);
    }

    function completeEscrow(uint256 escrowId) public onlySeller(escrowId) inStatus(escrowId, EscrowStatus.Funded) {
        escrows[escrowId].status = EscrowStatus.Completed;
        payable(escrows[escrowId].seller).transfer(escrows[escrowId].amount);
        emit EscrowCompleted(escrowId);
    }

    function disputeEscrow(uint256 escrowId) public onlyBuyer(escrowId) inStatus(escrowId, EscrowStatus.Funded) {
        escrows[escrowId].status = EscrowStatus.Disputed;
        emit EscrowDisputed(escrowId);
    }

    function resolveDispute(uint256 escrowId, bool releaseToSeller) public onlyArbiter(escrowId) inStatus(escrowId, EscrowStatus.Disputed) {
        if (releaseToSeller) {
            escrows[escrowId].status = EscrowStatus.Completed;
            payable(escrows[escrowId].seller).transfer(escrows[escrowId].amount);
            emit EscrowCompleted(escrowId);
        } else {
            escrows[escrowId].status = EscrowStatus.Refunded;
            payable(escrows[escrowId].buyer).transfer(escrows[escrowId].amount);
            emit EscrowRefunded(escrowId);
        }
    }

    function getEscrowDetails(uint256 escrowId) public view returns (address buyer, address seller, address arbiter, uint256 amount, EscrowStatus status) {
        Escrow memory escrow = escrows[escrowId];
        return (escrow.buyer, escrow.seller, escrow.arbiter, escrow.amount, escrow.status);
    }
}
