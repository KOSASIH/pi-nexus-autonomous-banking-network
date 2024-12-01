// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EscrowContract {
    enum EscrowState { Created, Funded, Completed, Canceled, Disputed, Resolved }

    struct Escrow {
        address buyer;
        address seller;
        address arbiter; // Third party for dispute resolution
        uint256 amount;
        uint256 createdAt;
        uint256 releaseTime; // Time after which funds can be released
        EscrowState state;
        bool isDisputed;
        uint256 interestRate; // Interest rate in basis points
    }

    mapping(uint256 => Escrow) public escrows;
    uint256 public escrowCounter;

    event EscrowCreated(uint256 indexed escrowId, address indexed buyer, address indexed seller, uint256 amount, uint256 releaseTime);
    event EscrowFunded(uint256 indexed escrowId);
    event EscrowCompleted(uint256 indexed escrowId);
    event EscrowCanceled(uint256 indexed escrowId);
    event EscrowDisputed(uint256 indexed escrowId);
    event EscrowResolved(uint256 indexed escrowId, address indexed winner);

    modifier onlyBuyer(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].buyer, "Only the buyer can call this function");
        _;
    }

    modifier onlySeller(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].seller, "Only the seller can call this function");
        _;
    }

    modifier onlyArbiter(uint256 escrowId) {
        require(msg.sender == escrows[escrowId].arbiter, "Only the arbiter can call this function");
        _;
    }

    modifier inState(uint256 escrowId, EscrowState state) {
        require(escrows[escrowId].state == state, "Invalid escrow state");
        _;
    }

    // Function to create a new escrow
    function createEscrow(address _seller, address _arbiter, uint256 _releaseTime, uint256 _interestRate) public payable {
        require(msg.value > 0, "Escrow amount must be greater than zero");
        require(_seller != msg.sender, "Buyer and seller cannot be the same");
        require(_arbiter != address(0), "Arbiter address cannot be zero");

        escrows[escrowCounter] = Escrow({
            buyer: msg.sender,
            seller: _seller,
            arbiter: _arbiter,
            amount: msg.value,
            createdAt: block.timestamp,
            releaseTime: _releaseTime,
            state: EscrowState.Created,
            isDisputed: false,
            interestRate: _interestRate
        });

        emit EscrowCreated(escrowCounter, msg.sender, _seller, msg.value, _releaseTime);
        escrowCounter++;
    }

    // Function to fund the escrow
    function fundEscrow(uint256 escrowId) public payable onlyBuyer(escrowId) inState(escrowId, EscrowState.Created) {
        require(msg.value > 0, "Funding amount must be greater than zero");
        require(msg.value == escrows[escrowId].amount, "Funding amount must match the escrow amount");

        escrows[escrowId].state = EscrowState.Funded;
        emit EscrowFunded(escrowId);
    }

    // Function to complete the escrow
    function completeEscrow(uint256 escrowId) public onlySeller(escrowId) inState(escrowId, EscrowState.Funded) {
        require(block.timestamp >= escrows[escrowId].releaseTime, "Cannot complete escrow before release time");

        escrows[escrowId].state = EscrowState.Completed;

        // Transfer funds to the seller
        payable(escrows[escrowId].seller).transfer(escrows[escrowId].amount);
        emit EscrowCompleted(escrowId);
    }

    // Function to cancel the escrow
    function cancelEscrow(uint256 escrowId) public onlyBuyer(escrowId) inState(escrowId, EscrowState.Created) {
        escrows[escrowId].state = EscrowState.Canceled;

        // Refund the buyer
        payable(esc rows[escrowId].buyer).transfer(escrows[escrowId].amount);
        emit EscrowCanceled(escrowId);
    }

    // Function to dispute the escrow
    function disputeEscrow(uint256 escrowId) public onlyBuyer(escrowId) inState(escrowId, EscrowState.Funded) {
        escrows[escrowId].isDisputed = true;
        escrows[escrowId].state = EscrowState.Disputed;
        emit EscrowDisputed(escrowId);
    }

    // Function for the arbiter to resolve the dispute
    function resolveDispute(uint256 escrowId, address winner) public onlyArbiter(escrowId) inState(escrowId, EscrowState.Disputed) {
        require(winner == escrows[escrowId].buyer || winner == escrows[escrowId].seller, "Winner must be either buyer or seller");

        escrows[escrowId].state = EscrowState.Resolved;
        escrows[escrowId].isDisputed = false;

        // Transfer funds to the winner
        payable(winner).transfer(escrows[escrowId].amount);
        emit EscrowResolved(escrowId, winner);
    }

    // Function to calculate interest (if applicable)
    function calculateInterest(uint256 escrowId) public view returns (uint256) {
        Escrow memory escrow = escrows[escrowId];
        if (escrow.state == EscrowState.Completed || escrow.state == EscrowState.Canceled) {
            return 0;
        }
        uint256 duration = block.timestamp - escrow.createdAt;
        uint256 interest = (escrow.amount * escrow.interestRate * duration) / (10000 * 365 days); // Interest in basis points
        return interest;
    }

    // Function to get escrow details
    function getEscrowDetails(uint256 escrowId) public view returns (address, address, address, uint256, uint256, uint256, EscrowState, bool, uint256) {
        Escrow memory escrow = escrows[escrowId];
        return (escrow.buyer, escrow.seller, escrow.arbiter, escrow.amount, escrow.createdAt, escrow.releaseTime, escrow.state, escrow.isDisputed, escrow.interestRate);
    }
}
