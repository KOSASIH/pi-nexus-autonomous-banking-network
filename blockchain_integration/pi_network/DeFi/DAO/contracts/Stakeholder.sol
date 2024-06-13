pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract Stakeholder {
    using SafeERC20 for IERC20;
    using SafeMath for uint256;
    using Address for address;

    // Mapping of stakeholders to their respective balances
    mapping (address => uint256) public stakeholderBalances;

    // Mapping of stakeholders to their respective reputation scores
    mapping (address => uint256) public stakeholderReputation;

    // Mapping of stakeholders to their respective voting power
    mapping (address => uint256) public stakeholderVotingPower;

    // Total supply of tokens
    uint256 public totalSupply;

    // Token contract instance
    IERC20 public tokenContract;

    // Reputation contract instance
    Reputation public reputationContract;

    // Event emitted when a stakeholder's balance changes
    event BalanceChanged(address indexed stakeholder, uint256 newBalance);

    // Event emitted when a stakeholder's reputation score changes
    event ReputationChanged(address indexed stakeholder, uint256 newReputation);

    // Event emitted when a stakeholder's voting power changes
    event VotingPowerChanged(address indexed stakeholder, uint256 newVotingPower);

    // Event emitted when a proposal is created
    event ProposalCreated(uint256 proposalId, address indexed proposer, string description);

    // Event emitted when a proposal is voted on
    event ProposalVoted(uint256 proposalId, address indexed voter, bool vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 proposalId, address indexed executor);

    // Struct to represent a proposal
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 votingDeadline;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    // Mapping of proposal IDs to proposal structs
    mapping (uint256 => Proposal) public proposals;

    // Next proposal ID
    uint256 public nextProposalId;

    // Constructor function
    constructor(IERC20 _tokenContract, Reputation _reputationContract) public {
        tokenContract = _tokenContract;
        reputationContract = _reputationContract;
        totalSupply = tokenContract.totalSupply();
    }

    // Function to stake tokens and update stakeholder's balance and reputation
    function stake(uint256 amount) public {
        require(amount > 0, "Amount must be greater than 0");
        tokenContract.safeTransferFrom(msg.sender, address(this), amount);
        stakeholderBalances[msg.sender] = stakeholderBalances[msg.sender].add(amount);
        totalSupply = totalSupply.add(amount);
        reputationContract.updateReputation(msg.sender, amount);
        emit BalanceChanged(msg.sender, stakeholderBalances[msg.sender]);
        emit ReputationChanged(msg.sender, reputationContract.getReputation(msg.sender));
    }

    // Function to unstake tokens and update stakeholder's balance and reputation
    function unstake(uint256 amount) public {
        require(amount > 0, "Amount must be greater than 0");
        require(stakeholderBalances[msg.sender] >= amount, "Insufficient balance");
        tokenContract.safeTransfer(msg.sender, amount);
        stakeholderBalances[msg.sender] = stakeholderBalances[msg.sender].sub(amount);
        totalSupply = totalSupply.sub(amount);
        reputationContract.updateReputation(msg.sender, -amount);
        emit BalanceChanged(msg.sender, stakeholderBalances[msg.sender]);
        emit ReputationChanged(msg.sender, reputationContract.getReputation(msg.sender));
    }

    // Function to create a new proposal
    function createProposal(string memory description) public {
        require(stakeholderBalances[msg.sender] > 0, "Only stakeholders can create proposals");
        Proposal memory newProposal = Proposal(nextProposalId, msg.sender, description, block.timestamp + 30 days, 0, 0, false);
        proposals[nextProposalId] = newProposal;
        emit ProposalCreated(nextProposalId, msg.sender, description);
        nextProposalId++;
    }

    // Function to vote on a proposal
    function vote(uint256 proposalId, bool vote) public {
        require(stakeholderBalances[msg.sender] > 0, "Only stakeholders can vote");
        require(proposals[proposalId].votingDeadline > block.timestamp, "Voting deadline has passed");
        if (vote) {
            proposals[proposalId].yesVotes = proposals[proposalId].yesVotes.add(stakeholderVotingPower[msg.sender]);
        } else {
            proposals[proposalId].noVotes = proposals[proposalId].noVotes.add(stakeholderVotingPower[msg.sender]);
        }
        emit ProposalVoted(proposalId, msg.sender, vote);
    }

    // Function to execute a proposal
    function executeProposal(uint256 proposalId) public {
        require(stakeholderBalances[msg.sender] > 0, "Only stakeholders can execute proposals");
        require(proposals[proposalId].executed == false, "Proposal already executed");
        require(block.timestamp > proposals[proposalId].votingDeadline, "Voting deadline not passed");
        require(proposals[proposalId].yesVotes > proposals[proposalId].noVotes, "Proposal not approved");
        // Execute proposal logic here
        proposals[proposalId].executed = true;
        emit ProposalExecuted(proposalId, msg.sender);
    }

    // Function to update stakeholder's voting power
    function updateVotingPower() public {
        stakeholderVotingPower[msg.sender] = stakeholderBalances[msg.sender].mul(reputationContract.getReputation(msg.sender));
    }

    // Function to get the total number of proposals
    function getProposalCount() public view returns (uint256) {
        return nextProposalId;
    }

    // Function to get a proposal by ID
    function getProposal(uint256 proposalId) public view returns (address, string memory, uint256, uint256, uint256, uint256, bool) {
        return (proposals[proposalId].proposer, proposals[proposalId].description, proposals[proposalId].votingDeadline, proposals[proposalId].yesVotes, proposals[proposalId].noVotes, proposals[proposalId].executed);
    }
}
