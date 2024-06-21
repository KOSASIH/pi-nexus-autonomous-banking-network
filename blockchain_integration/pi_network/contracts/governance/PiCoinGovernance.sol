// PiCoinGovernance.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiCoinGovernance {
    using SafeERC20 for IERC20;

    // Mapping of proposals to their vote counts
    mapping (bytes32 => uint256) public proposalVotes;

    // Event emitted when a new proposal is created
    event NewProposal(bytes32 proposalId, string description);

    // Event emitted when a proposal is voted on
    event ProposalVoted(bytes32 proposalId, address voter, bool inFavor);

    // Function to create a new proposal
    function createProposal(string memory description) public {
        bytes32 proposalId = keccak256(abi.encodePacked(description));
        proposalVotes[proposalId] = 0;
        emit NewProposal(proposalId, description);
    }

    // Function to vote on a proposal
    function voteOnProposal(bytes32 proposalId, bool inFavor) public {
        require(proposalVotes[proposalId] > 0, "Proposal does not exist");
        proposalVotes[proposalId] += inFavor ? 1 : -1;
        emit ProposalVoted(proposalId, msg.sender, inFavor);
    }
}
