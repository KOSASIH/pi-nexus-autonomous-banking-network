pragma solidity ^0.8.0;

contract CognitaDAO {
    // Mapping of proposals
    mapping (uint => Proposal) public proposals;

    // Mapping of votes
    mapping (address => mapping (uint => bool)) public votes;

    // Treasury balance
    uint public treasuryBalance;

    // Event emitted when a proposal is created
    event ProposalCreated(uint indexed proposalId, string description);

    // Event emitted when a vote is cast
    event VoteCast(address indexed voter, uint indexed proposalId, bool vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint indexed proposalId);

    // Constructor function
    constructor() public {
        treasuryBalance = 0;
    }

    // Function to create a proposal
    function createProposal(string memory description) public {
        uint proposalId = proposals.length++;
        proposals[proposalId] = Proposal(description, 0, 0);
        emit ProposalCreated(proposalId, description);
    }

    // Function to cast a vote
    function castVote(uint proposalId, bool vote) public {
        require(votes[msg.sender][proposalId] == false, "Already voted");
        votes[msg.sender][proposalId] = true;
        if (vote) {
            proposals[proposalId].yesVotes++;
        } else {
            proposals[proposalId].noVotes++;
        }
        emit VoteCast(msg.sender, proposalId, vote);
    }

    // Function to execute a proposal
    function executeProposal(uint proposalId) public {
        require(proposals[proposalId].yesVotes > proposals[proposalId].noVotes, "Proposal did not pass");
        // Execute proposal logic
        treasuryBalance += proposals[proposalId].value;
        emit ProposalExecuted(proposalId);
    }
}

struct Proposal {
    string description;
    uint yesVotes;
    uint noVotes;
    uint value;
}
