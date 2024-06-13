pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract DAO {
    using SafeERC20 for IERC20;

    // Mapping of proposals to votes
    mapping (address => mapping (uint256 => uint256)) public votes;

    // Mapping of proposals to descriptions
    mapping (uint256 => string) public proposalDescriptions;

    // Mapping of users to their balances
    mapping (address => uint256) public balances;

    // Event emitted when a new proposal is created
    event NewProposal(uint256 proposalId, string description);

    // Event emitted when a user votes on a proposal
    event VoteCast(address user, uint256 proposalId, uint256 vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId);

    // Function to create a new proposal
    function createProposal(string memory description) public {
        // Generate a unique proposal ID
        uint256 proposalId = uint256(keccak256(abi.encodePacked(block.timestamp, msg.sender)));

        // Set the proposal description
        proposalDescriptions[proposalId] = description;

        // Emit the NewProposal event
        emit NewProposal(proposalId, description);
    }

    // Function to vote on a proposal
    function vote(uint256 proposalId, uint256 vote) public {
        // Check if the user has already voted on this proposal
        require(votes[msg.sender][proposalId] == 0, "User has already voted on this proposal");

        // Update the user's vote
        votes[msg.sender][proposalId] = vote;

        // Emit the VoteCast event
        emit VoteCast(msg.sender, proposalId, vote);
    }

    // Function to execute a proposal
    function executeProposal(uint256 proposalId) public {
        // Check if the proposal has been approved
        require(getProposalApproval(proposalId) > 50, "Proposal has not been approved");

        // Execute the proposal
        // TO DO: implement proposal execution logic

        // Emit the ProposalExecuted event
        emit ProposalExecuted(proposalId);
    }

    // Function to get the approval percentage of a proposal
    function getProposalApproval(uint256 proposalId) public view returns (uint256) {
        // Calculate the total votes
        uint256 totalVotes = 0;
        for (address user in votes) {
            totalVotes += votes[user][proposalId];
        }

        // Calculate the approval percentage
        uint256 approvalPercentage = (totalVotes / balances[msg.sender]) * 100;

        return approvalPercentage;
    }
}
