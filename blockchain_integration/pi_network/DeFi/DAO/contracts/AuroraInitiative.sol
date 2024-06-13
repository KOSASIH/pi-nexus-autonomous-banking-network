pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract AuroraInitiative {
    using SafeERC20 for IERC20;
    using Counters for Counters.Counter;
    using SafeMath for uint256;

    // Events
    eventNewProposal(address indexed proposer, uint256 proposalId);
    eventVoteCast(address indexed voter, uint256 proposalId, bool support);
    eventProposalExecuted(uint256 proposalId);
    eventNewMember(address indexed member);
    eventMemberRemoved(address indexed member);
    eventReputationUpdated(address indexed member, uint256 newReputation);
    eventTokenTransferred(address indexed recipient, uint256 amount);
    eventAuroraUpdated(uint256 auroraId, uint256 auroraValue);
    eventAuroraReward(address indexed recipient, uint256 rewardAmount);

    // Structs
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 startTime;
        uint256 endTime;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
        bytes data; // additional data for proposal execution
    }

    struct Member {
        address member;
        uint256 joinedAt;
        uint256 reputation;
        uint256 tokenBalance;
        uint256 auroraValue; // aurora value of the member
        address[] auroraNodes; // list of aurora nodes connected to this member
    }

    struct AuroraNode {
        address node;
        uint256 connectedAt;
        uint256 auroraValue; // aurora value of the node
    }

    struct Aurora {
        uint256 id;
        uint256 auroraValue;
        uint256 rewardAmount;
        uint256 timestamp;
    }

    // Mappings
    mapping (address => Member) public members;
    mapping (uint256 => Proposal) public proposals;
    mapping (address => mapping (uint256 => bool)) public votes;
    mapping (address => uint256) public tokenBalances;
    mapping (address => AuroraNode) public auroraNodes;
    mapping (uint256 => Aurora) public auroras;

    // Counters
    Counters.Counter public proposalCount;
    Counters.Counter public memberCount;
    Counters.Counter public auroraNodeCount;
    Counters.Counter public auroraCount;

    // Constants
    uint256 public constant MIN_PROPOSAL_QUORUM = 50; // 50% of total members
    uint256 public constant VOTE_DURATION = 7 days;
    uint256 public constant PROPOSAL_EXECUTION_DELAY = 3 days;
    uint256 public constant REPUTATION_INCREMENT = 10; // reputation increment for voting
    uint256 public constant TOKEN_SUPPLY = 1000000; // initial token supply
    uint256 public constant AURORA_REWARD_INTERVAL = 1 hours; // interval for aurora rewards
    uint256 public constant AURORA_REWARD_AMOUNT = 100; // amount of tokens rewarded for each aurora

    // Modifiers
    modifier onlyMember() {
        require(members[msg.sender].member!= address(0), "Only members can call this function");
        _;
    }

    modifier onlyProposer(uint256 proposalId) {
        require(proposals[proposalId].proposer == msg.sender, "Only the proposer can call this function");
        _;
    }

    modifier onlyAuroraNode(address node) {
        require(auroraNodes[node].node!= address(0), "Only aurora nodes can call this function");
        _;
    }

    // Functions
    constructor() public {
        // Initialize the DAO with the creator as the first member
        members[msg.sender] = Member(msg.sender, block.timestamp, 0, TOKEN_SUPPLY, 0, new address[](0));
        memberCount.increment();
        tokenBalances[msg.sender] = TOKEN_SUPPLY;
    }

    function propose(string memory description, bytes memory data) public onlyMember {
        // Create a new proposal
        uint256 proposalId = proposalCount.current();
        proposals[proposalId] = Proposal(proposalId, msg.sender, description, block.timestamp, block.timestamp + VOTE_DURATION, 0, 0, false, data);
        proposalCount.increment();

        emit NewProposal(msg.sender, proposalId);
    }

    function vote(uint256proposalId, bool support) public onlyMember {
        // Check if the proposal exists and is still open
        require(proposals[proposalId].endTime > block.timestamp, "Proposal is no longer open");

        // Check if the member has already voted
        require(!votes[msg.sender][proposalId], "Member has already voted");

        // Update the vote count
        if (support) {
            proposals[proposalId].yesVotes++;
        } else {
            proposals[proposalId].noVotes++;
        }

        votes[msg.sender][proposalId] = true;

        // Update member reputation
        members[msg.sender].reputation += REPUTATION_INCREMENT;

        emit VoteCast(msg.sender, proposalId, support);
        emit ReputationUpdated(msg.sender, members[msg.sender].reputation);
    }

    function executeProposal(uint256 proposalId) public onlyProposer(proposalId) {
        // Check if the proposal has reached quorum and is still open
        require(proposals[proposalId].endTime < block.timestamp, "Proposal is still open");
        require(proposals[proposalId].yesVotes >= MIN_PROPOSAL_QUORUM, "Proposal did not reach quorum");

        // Execute the proposal
        // TO DO: implement proposal execution logic
        // For example, transfer tokens to a specified address
        IERC20(proposals[proposalId].data[0]).safeTransfer(address(proposals[proposalId].data[1]), uint256(proposals[proposalId].data[2]));

        proposals[proposalId].executed = true;

        emit ProposalExecuted(proposalId);
    }

    function joinDAO() public {
        // Check if the member is not already part of the DAO
        require(members[msg.sender].member == address(0), "Member is already part of the DAO");

        // Add the member to the DAO
        members[msg.sender] = Member(msg.sender, block.timestamp, 0, 0, 0, new address[](0));
        memberCount.increment();

        emit NewMember(msg.sender);
    }

    function leaveDAO() public onlyMember {
        // Remove the member from the DAO
        delete members[msg.sender];
        memberCount.decrement();

        emit MemberRemoved(msg.sender);
    }

    function transferTokens(address recipient, uint256 amount) public onlyMember {
        // Transfer tokens to another member
        require(tokenBalances[msg.sender] >= amount, "Insufficient token balance");
        require(recipient!= address(0), "Invalid recipient address");

        tokenBalances[msg.sender] = tokenBalances[msg.sender].sub(amount);
        tokenBalances[recipient] = tokenBalances[recipient].add(amount);

        emit TokenTransferred(recipient, amount);
    }

    function updateReputation(address member, uint256 newReputation) public onlyMember {
        // Update the reputation of another member
        require(member!= msg.sender, "Cannot update own reputation");
        require(newReputation >= 0, "Invalid reputation value");

        members[member].reputation = newReputation;

        emit ReputationUpdated(member, newReputation);
    }

    function addAuroraNode(address node) public onlyMember {
        // Add a new aurora node to the DAO
        require(auroraNodes[node].node == address(0), "Aurora node already exists");
        require(node!= address(0), "Invalid aurora node address");

        auroraNodes[node] = AuroraNode(node, block.timestamp, 0);
        auroraNodeCount.increment();

        emit AuroraNodeAdded(node);
    }

    function removeAuroraNode(address node) public onlyMember {
        // Remove an aurora node from the DAO
        require(auroraNodes[node].node!= address(0), "Aurora node does not exist");

        delete auroraNodes[node];
        auroraNodeCount.decrement();

        emit AuroraNodeRemoved(node);
    }

    function updateAuroraValue(address node, uint256 newAuroraValue) public onlyAuroraNode(node) {
        // Update the aurora value of an aurora node
        require(node!= address(0), "Invalid aurora node address");
        require(newAuroraValue >= 0, "Invalid aurora value");

        auroraNodes[node].auroraValue = newAuroraValue;

        emit AuroraUpdated(node, newAuroraValue);
    }

    function aurora() public {
        // Update the aurora value of all aurora nodes
        for (address node in auroraNodes) {
            auroraNodes[node].auroraValue += 1;
        }

        // Reward aurora nodes with tokens
        for (address node in auroraNodes) {
            tokenBalances[node] += AURORA_REWARD_AMOUNT;
            emit AuroraReward(node, AURORA_REWARD_AMOUNT);
        }

        // Create a new aurora
        uint256 auroraId = auroraCount.current();
        auroras[auroraId] = Aurora(auroraId, block.timestamp, AURORA_REWARD_AMOUNT);
        auroraCount.increment();

        emit AuroraUpdated(auroraId, AURORA_REWARD_AMOUNT);
    }

// Utility functions
    function getProposal(uint256 proposalId) public view returns (Proposal memory) {
        return proposals[proposalId];
    }

    function getMember(address member) public view returns (Member memory) {
        return members[member];
    }

    function getAuroraNode(address node) public view returns (AuroraNode memory) {
        return auroraNodes[node];
    }

    function getAurora(uint256 auroraId) public view returns (Aurora memory) {
        return auroras[auroraId];
    }

    function getMemberCount() public view returns (uint256) {
        return memberCount.current();
    }

    function getProposalCount() public view returns (uint256) {
        return proposalCount.current();
    }

    function getAuroraNodeCount() public view returns (uint256) {
        return auroraNodeCount.current();
    }

    function getAuroraCount() public view returns (uint256) {
        return auroraCount.current();
    }
}
