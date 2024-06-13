pragma solidity ^0.8.0;

interface IDAO {
    // Function to create a new proposal
    function createProposal(string memory description) external returns (uint256);

    // Function to vote on a proposal
    function vote(uint256 proposalId, uint256 vote) external;

    // Function to execute a proposal
    function executeProposal(uint256 proposalId) external;

    // Function to get the approval percentage of a proposal
    function getProposalApproval(uint256 proposalId) external view returns (uint256);

    // Function to get the balance of a user
    function getBalance(address user) external view returns (uint256);

    // Function to transfer tokens
    function transfer(address from, address to, uint256 amount) external;

    // Event emitted when a new proposal is created
    event NewProposal(uint256 proposalId, string description);

    // Event emitted when a user votes on a proposal
    event VoteCast(address user, uint256 proposalId, uint256 vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId);
}
