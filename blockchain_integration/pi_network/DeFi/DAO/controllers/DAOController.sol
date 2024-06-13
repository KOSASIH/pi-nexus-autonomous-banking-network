pragma solidity ^0.8.0;

import "./DAOModel.sol";
import "./GovernanceController.sol";
import "./TokenController.sol";

contract DAOController {
    // DAO model instance
    DAOModel public daoModel;

    // Governance controller instance
    GovernanceController public governanceController;

    // Token controller instance
    TokenController public tokenController;

    // Event emitted when a new proposal is created
    event NewProposal(uint256 proposalId, string description);

    // Event emitted when a user votes on a proposal
    event VoteCast(address user, uint256 proposalId, uint256 vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId);

    // Constructor
    constructor(address daoModelAddress, address governanceControllerAddress, address tokenControllerAddress) public {
        daoModel = DAOModel(daoModelAddress);
        governanceController = GovernanceController(governanceControllerAddress);
        tokenController = TokenController(tokenControllerAddress);
    }

    // Function to create a new proposal
    function createProposal(string memory description) public {
        uint256 proposalId = daoModel.createProposal(description);
        emit NewProposal(proposalId, description);
    }

    // Function to vote on a proposal
    function vote(uint256 proposalId, uint256 vote) public {
        daoModel.vote(proposalId, vote);
        emit VoteCast(msg.sender, proposalId, vote);
    }

    // Function to execute a proposal
    function executeProposal(uint256 proposalId) public {
        daoModel.executeProposal(proposalId);
        emit ProposalExecuted(proposalId);
    }

    // Function to get the balance of a user
    function getBalance(address user) public view returns (uint256) {
        return daoModel.getBalance(user);
    }

    // Function to transfer tokens
    function transfer(address from, address to, uint256 amount) public {
        daoModel.transfer(from, to, amount);
    }
}
