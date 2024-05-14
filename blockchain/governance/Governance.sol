// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/GovernorBravo.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/compatibility/GovernorCompatibilityBravo.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorVotes.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorTimelockControl.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorVeto.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorPendingAdmin.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorTermLimit.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorPendingTimelockControl.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorPendingVeto.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorTermLimitControl.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorConsecutiveVotes.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorState.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorDelayedRecipient.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalThreshold.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalEta.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalPowers.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalCancel.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalQueue.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalHash.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalCountdown.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalSnapshot.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMetadata.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalExecutor.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalVotingDelay.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalVotingPeriod.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMinQuorum.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalVetoPeriod.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxOperations.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxValue.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGas.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxOperationsPerEta.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxValuePerEta.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerOperation.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerEta.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxOperationsPerEtaPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxValuePerEtaPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerOperationPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerEtaPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxOperationsPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxValuePerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerOperationPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxGasPerAddress.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxOperationsPerEtaPerVoter.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/extensions/GovernorProposalMaxValuePerEtaPerVoter.sol";

contract Governance is Ownable, GovernorBravo, GovernorCompatibilityBravo, GovernorVotes, GovernorTimelockControl, GovernorVeto, GovernorPendingAdmin, GovernorTermLimit, GovernorTermLimitControl, GovernorConsecutiveVotes, GovernorState, GovernorDelayedRecipient, GovernorProposalThreshold, GovernorProposalEta, GovernorProposalPowers, GovernorProposalCancel, GovernorProposalQueue, GovernorProposalHash, GovernorProposalCountdown, GovernorProposalSnapshot, GovernorProposalMetadata, GovernorProposalExecutor, GovernorProposalVotingDelay, GovernorProposalVotingPeriod, GovernorProposalMinQuorum, GovernorProposalVetoPeriod, GovernorProposalMaxOperations, GovernorProposalMaxValue, GovernorProposalMaxGas, GovernorProposalMaxOperationsPerEta, GovernorProposalMaxValuePerEta, GovernorProposalMaxGasPerOperation, GovernorProposalMaxGasPerEta, GovernorProposalMaxOperationsPerEtaPerAddress, GovernorProposalMaxValuePerEtaPerAddress, GovernorProposalMaxGasPerOperationPerAddress, GovernorProposalMaxGasPerEtaPerAddress, GovernorProposalMaxOperationsPerAddress, GovernorProposalMaxValuePerAddress, GovernorProposalMaxGasPerOperationPerAddress, GovernorProposalMaxGasPerAddress, GovernorProposalMaxOperationsPerEtaPerVoter, GovernorProposalMaxValuePerEtaPerVoter {
    // Additional state variables and functions can be added here
}
