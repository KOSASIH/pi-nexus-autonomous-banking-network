pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PulsarPlatform {
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
    eventPulseUpdated(uint256 pulseId, uint256 pulseValue);
    eventPulseReward(address indexed recipient, uint256 rewardAmount);

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
        uint256 pulseValue; // pulse value of the member
        address[] pulseNodes; // list of pulse nodes connected to this member
    }

    struct PulseNode {
        address node;
        uint256 connectedAt;
        uint256 pulseValue; // pulse value of the node
    }

    struct Pulse {
        uint256 id;
        uint256 pulseValue;
        uint256 rewardAmount;
        uint256 timestamp;
    }

    // Mappings
    mapping (address => Member) public members;
    mapping (uint256 => Proposal) public proposals;
    mapping (address => mapping (uint256 => bool)) public votes;
    mapping (address => uint256) public tokenBalances;
    mapping (address => PulseNode) public pulseNodes;
    mapping (uint256 => Pulse) public pulses;

    // Counters
    Counters.Counter public proposalCount;
    Counters.Counter public memberCount;
    Counters.Counter public pulseNodeCount;
    Counters.Counter public pulseCount;

    // Constants
    uint256 public constant MIN_PROPOSAL_QUORUM = 50; // 50% of total members
    uint256 public constant VOTE_DURATION = 7 days;
    uint256 public constant PROPOSAL_EXECUTION_DELAY = 3 days;
    uint256 public constant REPUTATION_INCREMENT = 10; // reputation increment for voting
    uint256 public constant TOKEN_SUPPLY = 1000000; // initial token supply
    uint256 public constant PULSE_REWARD_INTERVAL = 1 hours; // interval for pulse rewards
    uint256 public constant PULSE_REWARD_AMOUNT = 100; // amount of tokens rewarded for each pulse

    // Modifiers
    modifier onlyMember() {
        require(members[msg.sender].member!= address(0), "Only members can call this function");
        _;
    }

    modifier onlyProposer(uint256 proposalId) {
        require(proposals[proposalId].proposer == msg.sender, "Only the proposer can call this function");
        _;
    }

    modifier onlyPulseNode(address node) {
        require(pulseNodes[node].node!= address(0), "Only pulse nodes can call this function");
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

    function vote(uint256 proposalId, bool support) public onlyMember {
        // Check if the proposal exists and is stillopen
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

    function addPulseNode(address node) public onlyMember {
        // Add a new pulse node to the DAO
        require(pulseNodes[node].node == address(0), "Pulse node already exists");
        require(node!= address(0), "Invalid pulse node address");

        pulseNodes[node] = PulseNode(node, block.timestamp, 0);
        pulseNodeCount.increment();

        emit PulseNodeAdded(node);
    }

    function removePulseNode(address node) public onlyMember {
        // Remove a pulse node from the DAO
        require(pulseNodes[node].node!= address(0), "Pulse node does not exist");

        delete pulseNodes[node];
        pulseNodeCount.decrement();

        emit PulseNodeRemoved(node);
    }

    function updatePulseValue(address node, uint256 newPulseValue) public onlyPulseNode(node) {
        // Update the pulse value of a pulse node
        require(node!= address(0), "Invalid pulse node address");
        require(newPulseValue >= 0, "Invalid pulse value");

        pulseNodes[node].pulseValue = newPulseValue;

        emit PulseUpdated(node, newPulseValue);
    }

    function pulse() public {
        // Update the pulse value of all pulse nodes
        for (address node in pulseNodes) {
            pulseNodes[node].pulseValue += 1;
        }

        // Reward pulse nodes with tokens
        for (address node in pulseNodes) {
            tokenBalances[node] += PULSE_REWARD_AMOUNT;
            emit PulseReward(node, PULSE_REWARD_AMOUNT);
        }

        // Create a new pulse
        uint256 pulseId = pulseCount.current();
        pulses[pulseId] = Pulse(pulseId, block.timestamp, PULSE_REWARD_AMOUNT);
        pulseCount.increment();

        emit PulseUpdated(pulseId, PULSE_REWARD_AMOUNT);
    }

// Utility functions
    function getProposal(uint256 proposalId) public view returns (Proposal memory) {
        return proposals[proposalId];
    }

    function getMember(address member) public view returns (Member memory) {
        return members[member];
    }

    function getPulseNode(address node) public view returns (PulseNode memory) {
        return pulseNodes[node];
    }

    function getPulse(uint256 pulseId) public view returns (Pulse memory) {
        return pulses[pulseId];
    }

    function getMemberCount() public view returns (uint256) {
        return memberCount.current();
    }

    function getProposalCount() public view returns (uint256) {
        return proposalCount.current();
    }

    function getPulseNodeCount() public view returns (uint256) {
        return pulseNodeCount.current();
    }

    function getPulseCount() public view returns (uint256) {
        return pulseCount.current();
    }
}
