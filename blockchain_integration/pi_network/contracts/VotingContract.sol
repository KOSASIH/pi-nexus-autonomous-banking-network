// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract VotingContract is Ownable {
    struct Candidate {
        string name;
        uint256 voteCount;
    }

    struct Voter {
        bool hasVoted;
        uint256 voteIndex;
        address delegate;
    }

    mapping(uint256 => Candidate) public candidates;
    mapping(address => Voter) public voters;
    uint256 public candidatesCount;
    uint256 public totalVotes;
    IERC20 public votingToken;
    uint256 public votingEndTime;

    event CandidateAdded(uint256 indexed candidateId, string name);
    event Voted(address indexed voter, uint256 indexed candidateId);
    event VoteDelegated(address indexed voter, address indexed delegate);

    modifier onlyDuringVoting() {
        require(block.timestamp < votingEndTime, "Voting has ended.");
        _;
    }

    constructor(address _votingToken, uint256 _votingDuration) {
        votingToken = IERC20(_votingToken);
        votingEndTime = block.timestamp + _votingDuration;
    }

    function addCandidate(string memory name) public onlyOwner {
        candidatesCount++;
        candidates[candidatesCount] = Candidate(name, 0);
        emit CandidateAdded(candidatesCount, name);
    }

    function vote(uint256 candidateId) public onlyDuringVoting {
        Voter storage sender = voters[msg.sender];
        require(!sender.hasVoted, "You have already voted.");
        require(candidateId > 0 && candidateId <= candidatesCount, "Invalid candidate ID.");

        // Transfer voting tokens to the contract
        uint256 tokenBalance = votingToken.balanceOf(msg.sender);
        require(tokenBalance > 0, "You must hold voting tokens to vote.");
        votingToken.transferFrom(msg.sender, address(this), tokenBalance);

        sender.hasVoted = true;
        sender.voteIndex = candidateId;
        candidates[candidateId].voteCount += tokenBalance;
        totalVotes += tokenBalance;

        emit Voted(msg.sender, candidateId);
    }

    function delegateVote(address to) public onlyDuringVoting {
        Voter storage sender = voters[msg.sender];
        require(!sender.hasVoted, "You have already voted.");
        require(to != msg.sender, "You cannot delegate to yourself.");

        sender.delegate = to;
        emit VoteDelegated(msg.sender, to);
    }

    function getResults() public view returns (string memory winnerName, uint256 winnerVoteCount) {
        uint256 winningVoteCount = 0;
        for (uint256 i = 1; i <= candidatesCount; i++) {
            if (candidates[i].voteCount > winningVoteCount) {
                winningVoteCount = candidates[i].voteCount;
                winnerName = candidates[i].name;
            }
        }
        winnerVoteCount = winningVoteCount;
    }

    function withdrawTokens() public onlyOwner {
        require(block.timestamp > votingEndTime, "Voting is still ongoing.");
        uint256 balance = votingToken.balanceOf(address(this));
        require(balance > 0, "No tokens to withdraw.");
        votingToken.transfer(msg.sender, balance);
    }
}
