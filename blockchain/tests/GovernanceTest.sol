// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "truffle/files/openzeppelin/test/helpers/Assert.sol";
import "truffle/files/openzeppelin/test/helpers/ExpectedError.sol";
import "truffle/files/openzeppelin/test/helpers/SetupContext.sol";

import "../contracts/Governance.sol";

contract GovernanceTest is SetupContext {
    Governance public governance;

    function setUp() public {
        governance = new Governance();
    }

    function testMinQuorum() public {
        Assert.equal(governance.minQuorum(), 10, "Governance: minQuorum is incorrect");
    }

    function testVotingPeriod() public {
        Assert.equal(governance.votingPeriod(), 100, "Governance: votingPeriod is incorrect");
    }

    function testVotingDelay() public {
        Assert.equal(governance.votingDelay(), 50, "Governance: votingDelay is incorrect");
    }

    function testMaxOperations() public {
        Assert.equal(governance.maxOperations(), 1000, "Governance: maxOperations is incorrect");
    }

    function testMaxValue() public {
        Assert.equal(governance.maxValue(), 10000, "Governance: maxValue is incorrect");
    }

    function testMaxGas() public {
        Assert.equal(governance.maxGas(), 1000000, "Governance: maxGas is incorrect");
    }

    function testMaxOperationsPerEta() public {
        Assert.equal(governance.maxOperationsPerEta(), 100, "Governance: maxOperationsPerEta is incorrect");
    }

    function testMaxValuePerEta() public {
        Assert.equal(governance.maxValuePerEta(), 1000, "Governance: maxValuePerEta is incorrect");
    }

    function testMaxGasPerOperation() public {
        Assert.equal(governance.maxGasPerOperation(), 100000, "Governance: maxGasPerOperation is incorrect");
    }

    function testMaxGasPerEta() public {
        Assert.equal(governance.maxGasPerEta(), 1000000, "Governance: maxGasPerEta is incorrect");
    }

    function testCreateProposal() public {
        uint256 newMaxValue = 1000;
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = newMaxValue;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        Assert.equal(proposalId, 1, "Governance: createProposal returns incorrect proposalId");
    }

    function testGetProposal() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        Governance.Proposal memory proposal = governance.getProposal(proposalId);
        Assert.equal(proposal.id, proposalId, "Governance: getProposal returns incorrect id");
        Assert.equal(proposal.targets.length, 1, "Governance: getProposal returns incorrect targets length");
        Assert.equal(proposal.values.length, 1, "Governance: getProposal returns incorrect values length");
        Assert.equal(proposal.signatures.length, 1, "Governance: getProposal returns incorrect signatures length");
        Assert.equal(proposal.eta, 0, "Governance: getProposal returns incorrect eta");
        Assert.equal(proposal.startBlock, 0, "Governance: getProposal returns incorrect startBlock");
        Assert.equal(proposal.endBlock, 0, "Governance: getProposal returns incorrect endBlock");
        Assert.equal(proposal.forVotes, 0, "Governance: getProposal returns incorrect forVotes");
        Assert.equal(proposal.againstVotes, 0, "Governance: getProposal returns incorrect againstVotes");
        Assert.equal(proposal.canceled, false, "Governance: getProposal returns incorrect canceled");
        Assert.equal(proposal.executed, false, "Governance: getProposal returns incorrect executed");
    }

    function testQueueProposal() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.castVote(proposalId, true);
        uint256 eta = governance.queueProposal(proposalId);
        Assert.equal(eta, 100, "Governance: queueProposal returns incorrect eta");
    }

    function testExecuteProposal() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

       uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.castVote(proposalId, true);
        uint256 eta = governance.queueProposal(proposalId);
        governance.executeProposal(proposalId, eta);
        Assert.equal(governance.getProposal(proposalId).executed, true, "Governance: executeProposal does not execute proposal");
    }

    function testCancelProposal() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.cancelProposal(proposalId);
        Assert.equal(governance.getProposal(proposalId).canceled, true, "Governance: cancelProposal does not cancel proposal");
    }

    function testGetProposalOperations() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        bytes32[] memory proposalOperations = governance.getProposalOperations(proposalId);
        Assert.equal(proposalOperations.length, 1, "Governance: getProposalOperations returns incorrect operations length");
        Assert.equal(proposalOperations[0], operations[0], "Governance: getProposalOperations returns incorrect operations");
    }

    function testGetProposalValues() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        uint256[] memory proposalValues = governance.getProposalValues(proposalId);
        Assert.equal(proposalValues.length, 1, "Governance: getProposalValues returns incorrect values length");
        Assert.equal(proposalValues[0], values[0], "Governance: getProposalValues returns incorrect values");
    }

    function testGetProposalTargets() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        address[] memory proposalTargets = governance.getProposalTargets(proposalId);
        Assert.equal(proposalTargets.length, 1, "Governance: getProposalTargets returns incorrect targets length");
        Assert.equal(proposalTargets[0], targets[0], "Governance: getProposalTargets returns incorrect targets");
    }

    function testGetProposalSignatures() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        bytes4[] memory proposalSignatures = governance.getProposalSignatures(proposalId);
        Assert.equal(proposalSignatures.length, 1, "Governance: getProposalSignatures returns incorrect signatures length");
        Assert.equal(proposalSignatures[0], bytes4(keccak256("test")), "Governance: getProposalSignatures returns incorrect signatures");
    }

    function testGetProposalEta() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        uint256 proposalEta = governance.getProposalEta(proposalId);
        Assert.equal(proposalEta, 0, "Governance: getProposalEta returns incorrect eta");
    }

    function testGetProposalStartBlock() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        uint256 proposalStartBlock = governance.getProposalStartBlock(proposalId);
        Assert.equal(proposalStartBlock, 0, "Governance: getProposalStartBlock returns incorrect startBlock");
    }

    function testGetProposalEndBlock() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");

        uint256 proposalEndBlock = governance.getProposalEndBlock(proposalId);
        Assert.equal(proposalEndBlock, 0, "Governance: getProposalEndBlock returns incorrect endBlock");
    }

    function testGetProposalForVotes() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.castVote(proposalId, true);

        uint256 proposalForVotes = governance.getProposalForVotes(proposalId);
        Assert.equal(proposalForVotes, 1, "Governance: getProposalForVotes returns incorrect forVotes");
    }

    function testGetProposalAgainstVotes() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId =governance.createProposal(operations, values, targets, "Test Proposal");
        governance.castVote(proposalId, false);

        uint256 proposalAgainstVotes = governance.getProposalAgainstVotes(proposalId);
        Assert.equal(proposalAgainstVotes, 1, "Governance: getProposalAgainstVotes returns incorrect againstVotes");
    }

    function testGetProposalCanceled() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.cancelProposal(proposalId);

        bool proposalCanceled = governance.getProposalCanceled(proposalId);
        Assert.equal(proposalCanceled, true, "Governance: getProposalCanceled returns incorrect canceled");
    }

    function testGetProposalExecuted() public {
        bytes32[] memory operations = new bytes32[](1);
        uint256[] memory values = new uint256[](1);
        address[] memory targets = new address[](1);
        operations[0] = keccak256("test");
        values[0] = 1000;
        targets[0] = address(this);

        uint256 proposalId = governance.createProposal(operations, values, targets, "Test Proposal");
        governance.executeProposal(proposalId, 0);

        bool proposalExecuted = governance.getProposalExecuted(proposalId);
        Assert.equal(proposalExecuted, true, "Governance: getProposalExecuted returns incorrect executed");
    }
}
