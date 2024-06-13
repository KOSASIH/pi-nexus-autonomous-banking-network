pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract NexusInfinity {
    using SafeERC20 for address;
    using Counters for Counters.Counter;
    using Ownable for address;

    // Mapping of members to their balances
    mapping (address => uint256) public memberBalances;

    // Mapping of proposals to their details
    mapping (uint256 => Proposal) public proposals;

    // Counter for proposal IDs
    Counters.Counter public proposalIdCounter;

    // Event emitted when a new proposal is created
    event NewProposal(uint256 proposalId, address proposer, string description);

    // Event emitted when a proposal is voted on
    event VoteCast(uint256 proposalId, address voter, bool inFavor);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId, bool success);

    // Event emitted when a new member is added
    event NewMember(address member);

    // Event emitted when a member is removed
    event MemberRemoved(address member);

    // Struct to represent a proposal
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 voteCount;
        uint256 inFavorCount;
        bool executed;
    }

    // Function to create a new proposal
    function createProposal(string memory _description) public onlyOwner {
        uint256 proposalId = proposalIdCounter.current();
        proposals[proposalId] = Proposal(proposalId, msg.sender, _description, 0, 0, false);
        emit NewProposal(proposalId, msg.sender, _description);
        proposalIdCounter.increment();
    }

    // Function to vote on a proposal
    function vote(uint256 _proposalId, bool _inFavor) public {
        Proposal storage proposal = proposals[_proposalId];
        require(proposal.executed == false, "Proposal has already been executed");
        require(memberBalances[msg.sender] > 0, "You are not a member");
        proposal.voteCount++;
        if (_inFavor) {
            proposal.inFavorCount++;
        }
        emit VoteCast(_proposalId, msg.sender, _inFavor);
    }

    // Function to execute a proposal
    function executeProposal(uint256 _proposalId) public onlyOwner {
        Proposal storage proposal = proposals[_proposalId];
        require(proposal.executed == false, "Proposal has already been executed");
        require(proposal.voteCount > 0, "No votes have been cast");
        if (proposal.inFavorCount > proposal.voteCount / 2) {
            // Execute the proposal
            //...
            proposal.executed = true;
            emit ProposalExecuted(_proposalId, true);
        } else {
            // Reject the proposal
            proposal.executed = true;
            emit ProposalExecuted(_proposalId, false);
        }
    }

    // Function to add a new member
    function addMember(address _newMember) public onlyOwner {
        memberBalances[_newMember] = 1;
        emit NewMember(_newMember);
    }

    // Function to remove a member
    function removeMember(address _member) public onlyOwner {
        delete memberBalances[_member];
        emit MemberRemoved(_member);
    }

    // Function to transfer ownership
    function transferOwnership(address _newOwner) public onlyOwner {
        Ownable.transferOwnership(_newOwner);
    }
}
