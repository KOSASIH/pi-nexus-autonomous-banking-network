pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiGovernanceV2 {
    using SafeMath for uint256;

    // Mapping of proposals
    mapping (uint256 => Proposal) public proposals;

    // Event emitted when a new proposal is created
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);

    // Event emitted when a proposal is voted on
    event VoteCast(uint256 indexed proposalId, address indexed voter, uint256 vote);

    // Event emitted when a proposal is executed
    event ProposalExecuted(uint256 indexed proposalId);

    // Struct to represent a proposal
    struct Proposal {
        uint256 proposalId;
        address proposer;
        string description;
        uint256 voteCount;
        mapping (address => uint256) votes;
        bool executed;
    }

    // Function to create a new proposal
    function createProposal(string memory description) public {
        // Create a new proposal
        Proposal memory proposal = Proposal(proposals.length + 1, msg.sender, description, 0, Proposal.Votes({}), false);
        proposals[proposal.proposalId] = proposal;

        // Emit the ProposalCreated event
        emit ProposalCreated(proposal.proposalId, msg.sender, description);
    }

    // Function to vote on a proposal
    function voteOnProposal(uint256 proposalId, uint256 vote) public {
        // Get the proposal
        Proposal storage proposal = proposals[proposalId];

        // Check if0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract PiOracleV2 is Ownable {
    using SafeMath for uint256;

    struct Oracle {
        uint256 timestamp;
        uint256 price;
    }

    mapping(address => Oracle[]) public oracles;
    mapping(address => uint256) public latestPrice;

    event PriceUpdated(address indexed token, uint256 indexed price, uint256 indexed timestamp);

    function updatePrice(address token, uint256 price) public onlyOwner {
        Oracle[] storage tokenOracles = oracles[token];
        uint256 currentPrice = tokenOracles[tokenOracles.length - 1].price;

        if (tokenOracles.length == 0 || price != currentPrice) {
            tokenOracles.push(Oracle(block.timestamp, price));
            latestPrice[token] = price;
            emit PriceUpdated(token, price, block.timestamp);
        }
    }

    function getOracleCount(address token) public view returns (uint256) {
        return oracles[token].length;
    }

    function getOraclePrice(address token, uint256 index) public view returns (uint256) {
        Oracle[] storage tokenOracles = oracles[token];
        require(index < tokenOracles.length, "Invalid oracle index");
        return tokenOracles[index].price;
    }

    function getOracleTimestamp(address token, uint256 index) public view returns (uint256) {
        Oracle[] storage tokenOracles = oracles[token];
        require(index < tokenOracles.length, "Invalid oracle index");
        return tokenOracles[index].timestamp;
    }
}
