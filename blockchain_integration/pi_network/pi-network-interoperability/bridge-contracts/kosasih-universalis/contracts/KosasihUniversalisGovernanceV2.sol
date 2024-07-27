pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";
import "./KosasihUniversalisUtils.sol";
import "./KosasihUniversalisMath.sol";

contract KosasihUniversalisGovernanceV2 {
    address public kosasihUniversalisNexus;
    address public governanceToken;

    // Mapping of proposals to their respective data
    mapping(bytes32 => Proposal) public proposals;

    // Mapping of voters to their respective votes
    mapping(address => mapping(bytes32 => bool)) public votes;

    // Event emitted when a new proposal is created
    event ProposalCreated(bytes32 indexed _proposalId, address indexed _contractAddress, bytes _updateData);

    // Event emitted when a vote is cast
    event VoteCast(address indexed _voter, bytes32 indexed _proposalId, bool _vote);

    // Event emitted when a proposal is executed
    event UpdateExecuted(bytes32 indexed _proposalId, address indexed _contractAddress, bytes _updateData);

    struct Proposal {
        address contractAddress;
        bytes updateData;
        uint256 startTime;
        uint256 endTime;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    constructor(address _kosasihUniversalisNexus, address _governanceToken) public {
        kosasihUniversalisNexus = _kosasihUniversalisNexus;
        governanceToken = _governanceToken;
    }

    /**
     * @dev Creates a new proposal to update a contract.
     * @param _contractAddress The address of the contract to be updated.
     * @param _updateData The data to be used for the update.
     */
    function proposeUpdate(address _contractAddress, bytes _updateData) public {
        bytes32 proposalId = keccak256(abi.encodePacked(_contractAddress, _updateData));
        Proposal storage proposal = proposals[proposalId];
        proposal.contractAddress = _contractAddress;
        proposal.updateData = _updateData;
        proposal.startTime = block.timestamp;
        proposal.endTime = block.timestamp + 7 days; // 7-day voting period
        proposal.yesVotes = 0;
        proposal.noVotes = 0;
        proposal.executed = false;
        emit ProposalCreated(proposalId, _contractAddress, _updateData);
    }

    /**
     * @dev Casts a vote on a proposal.
     * @param _proposalId The ID of the proposal to vote on.
     * @param _vote The vote (true for yes, false for no).
     */
    function voteOnProposal(bytes32 _proposalId, bool _vote) public {
        Proposal storage proposal = proposals[_proposalId];
        require(proposal.startTime <= block.timestamp && proposal.endTime >= block.timestamp, "Voting period has ended");
        require(votes[msg.sender][_proposalId] == false, "You have already voted on this proposal");
        if (_vote) {
            proposal.yesVotes++;
        } else {
            proposal.noVotes++;
        }
        votes[msg.sender][_proposalId] = true;
        emit VoteCast(msg.sender, _proposalId, _vote);
    }

    /**
     * @dev Executes a proposal if it has reached the required threshold.
     * @param _proposalId The ID of the proposal to execute.
     */
    function executeProposal(bytes32 _proposalId) public {
        Proposal storage proposal = proposals[_proposalId];
        require(proposal.endTime <= block.timestamp, "Voting period has not ended");
        require(proposal.executed == false, "Proposal has already been executed");
        uint256 totalVotes = proposal.yesVotes + proposal.noVotes;
        require(totalVotes > 0, "No votes have been cast");
        uint256 threshold = totalVotes * 2 / 3; // 2/3 majority required
        if (proposal.yesVotes >= threshold) {
            // Execute the proposal
            (bool success, ) = proposal.contractAddress.call(proposal.updateData);
            require(success, "Proposal execution failed");
            proposal.executed = true;
            emit UpdateExecuted(_proposalId, proposal.contractAddress, proposal.updateData);
        }
    }
}
