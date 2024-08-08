pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiNetworkGovernanceContract {
    // Mapping of proposals to their details
    mapping (uint256 => Proposal) public proposals;

    // Mapping of voters to their votes
    mapping (address => mapping (uint256 => bool)) public votes;

    // Event emitted when a new proposal is created
    event NewProposal(uint256 proposalId, address proposer, string description);

    // Event emitted when a proposal is voted on
    event VoteCast(uint256 proposalId, address voter, bool vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId);

    // Struct to represent a proposal
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 voteCount;
        bool executed;
    }

    // Function to create a new proposal
    function createProposal(string memory _description) public {
        uint256 proposalId = proposals.length++;
        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.description = _description;
        proposal.voteCount = 0;
        proposal.executed = false;
        emit NewProposal(proposalId, msg.sender, _description);
    }

    // Function to vote on a proposal
    function vote(uint256 _proposalId, bool _vote) public {
        require(proposals[_proposalId].executed == false, "Proposal has already been executed");
        require(votes[msg.sender][_proposalId] == false, "You have already voted on this proposal");
        Proposal storage proposal = proposals[_proposalId];
        proposal.voteCount++;
        votes[msg.sender][_proposalId] = _vote;
        emit VoteCast(_proposalId, msg.sender, _vote);
        if (proposal.voteCount >= (proposal.voteCount / 2) + 1) {
            executeProposal(_proposalId);
        }
    }

    // Function to execute a proposal
    function executeProposal(uint256 _proposalId) internal {
        Proposal storage proposal = proposals[_proposalId];
        require(proposal.executed == false, "Proposal has already been executed");
        proposal.executed = true;
        emit ProposalExecuted(_proposalId);
        // Execute the proposal logic here
        // For example, you could call a function on the PiNetworkContract
        // PiNetworkContract(piNetworkContractAddress).executeProposal(_proposalId);
    }

    // Function to get the proposal count
    function getProposalCount() public view returns (uint256) {
        return proposals.length;
    }

    // Function to get a proposal by ID
    function getProposalById(uint256 _proposalId) public view returns (uint256, address, string memory, uint256, bool) {
        Proposal storage proposal = proposals[_proposalId];
        return (proposal.id, proposal.proposer, proposal.description, proposal.voteCount, proposal.executed);
    }
}
