// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract GovernanceContract is Ownable {
    using SafeMath for uint256;

    IERC20 public quantumCoin; // The QuantumCoin token contract

    struct Proposal {
        uint256 id; // Proposal ID
        string description; // Proposal description
        uint256 voteCountFor; // Votes in favor
        uint256 voteCountAgainst; // Votes against
        uint256 endTime; // Voting end time
        bool executed; // Whether the proposal has been executed
        mapping(address => bool) hasVoted; // Track if an address has voted
    }

    Proposal[] public proposals; // Array of proposals
    uint256 public proposalCount; // Total number of proposals

    event ProposalCreated(uint256 id, string description, uint256 endTime);
    event Voted(uint256 proposalId, address voter, bool support);
    event ProposalExecuted(uint256 proposalId);

    constructor(IERC20 _quantumCoin) {
        quantumCoin = _quantumCoin;
    }

    // Function to create a new proposal
    function createProposal(string memory description, uint256 duration) external onlyOwner {
        require(duration > 0, "Duration must be greater than 0");

        Proposal storage newProposal = proposals.push();
        newProposal.id = proposalCount;
        newProposal.description = description;
        newProposal.endTime = block.timestamp.add(duration);
        newProposal.executed = false;

        emit ProposalCreated(proposalCount, description, newProposal.endTime);
        proposalCount++;
    }

    // Function to vote on a proposal
    function vote(uint256 proposalId, bool support) external {
        require(proposalId < proposalCount, "Invalid proposal ID");
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp < proposal.endTime, "Voting has ended");
        require(!proposal.hasVoted[msg.sender], "You have already voted");

        // Record the vote
        proposal.hasVoted[msg.sender] = true;

        // Count the vote
        if (support) {
            proposal.voteCountFor = proposal.voteCountFor.add(quantumCoin.balanceOf(msg.sender));
        } else {
            proposal.voteCountAgainst = proposal.voteCountAgainst.add(quantumCoin.balanceOf(msg.sender));
        }

        emit Voted(proposalId, msg.sender, support);
    }

    // Function to execute a proposal if it has passed
    function executeProposal(uint256 proposalId) external {
        require(proposalId < proposalCount, "Invalid proposal ID");
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.endTime, "Voting is still ongoing");
        require(!proposal.executed, "Proposal has already been executed");

        // Check if the proposal has enough votes in favor
        require(proposal.voteCountFor > proposal.voteCountAgainst, "Proposal did not pass");

        // Execute the proposal (custom logic can be added here)
        // For example, you could change a state variable or call another contract

        proposal.executed = true;

        emit ProposalExecuted(proposalId);
    }

    // Function to get the details of a proposal
    function getProposal(uint256 proposalId) external view returns (
        uint256 id,
        string memory description,
        uint256 voteCountFor,
        uint256 voteCountAgainst,
        uint256 endTime,
        bool executed
    ) {
        require(proposalId < proposalCount, "Invalid proposal ID");
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.id,
            proposal.description,
            proposal.voteCountFor,
            proposal.voteCountAgainst,
            proposal.endTime,
            proposal.executed
        );
    }

    // Function to get the total number of proposals
    function getTotalProposals() external view returns (uint256) {
        return proposalCount;
    }
}
