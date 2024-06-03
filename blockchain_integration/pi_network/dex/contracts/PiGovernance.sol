pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiGovernance {
    using SafeMath for uint256;

    // Mapping of proposals
    mapping (uint256 => Proposal) public proposals;

    // Event emitted when a new proposal is created
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);

    // Event emitted when a proposal is voted on
    event VoteCast(uint256 indexed proposalId, address indexed voter, uint256 vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 indexed proposalId);

    // Struct to represent a proposal
    struct Proposal {
        uint256 proposalId;
        address proposer;
        string description;
        uint256 voteCount;
        mapping (address => uint256) votes;
    }

    // Function to create a new proposal
    function createProposal(string memory description) public {
        // Create a new proposal
        Proposal memory proposal = Proposal(proposals.length + 1, msg.sender, description, 0);
        proposals[proposal.proposalId] = proposal;

        // Emit the ProposalCreated event
        emit ProposalCreated(proposal.proposalId, msg.sender, description);
    }

    // Function to vote on a proposal
    function voteOnProposal(uint256 proposalId, uint256 vote) public {
        // Get the proposal
        Proposal storage proposal = proposals[proposalId];

        // Check if the voter has already voted
        if (proposal.votes[msg.sender]!= 0) revert("Voter has already voted");

        // Update the vote count and voter's vote
        proposal.voteCount = proposal.voteCount.add(vote);
        proposal.votes[msg.sender] = vote;

        // Emit the VoteCast event
        emit VoteCast(proposalId, msg.sender, vote);
    }

    // Function to execute a proposal
    function executeProposal(uint256 proposalId) public {
        // Get the proposal
        Proposal storage proposal = proposals[proposalId];

        // Check if the proposal has been voted on
        if (proposal.voteCount == 0) revert("Proposal has not been voted on");

        // Execute the proposal
        // TO DO: implement the proposal execution logic

        // Emit the ProposalExecuted event
        emit ProposalExecuted(proposalId);
    }
}
