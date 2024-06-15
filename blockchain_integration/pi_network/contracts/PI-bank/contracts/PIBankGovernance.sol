pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankGovernance {
    // Mapping of proposals
    mapping(uint256 => Proposal) public proposals;

    // Event
    event NewProposal(address indexed proposer, uint256 proposalId);

    // Function
    function propose(address proposer, uint256 proposalId) public {
        // Create a new proposal
        Proposal proposal = Proposal(proposer, proposalId);
        proposals[proposalId] = proposal;
        emit NewProposal(proposer, proposalId);
    }
}
