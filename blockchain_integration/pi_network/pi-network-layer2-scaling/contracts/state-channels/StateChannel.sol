// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./ChannelManager.sol";

contract StateChannel {
    ChannelManager public manager;
    address public participantA;
    address public participantB;
    uint256 public balanceA;
    uint256 public balanceB;
    bool public isOpen;

    event ChannelOpened(address indexed participantA, address indexed participantB);
    event ChannelClosed(address indexed participant);
    event BalanceUpdated(address indexed participant, uint256 newBalance);

    modifier onlyParticipants() {
        require(msg.sender == participantA || msg.sender == participantB, "Not a participant");
        _;
    }

    constructor(address _participantA, address _participantB, address _manager) {
        participantA = _participantA;
        participantB = _participantB;
        manager = ChannelManager(_manager);
        isOpen = true;
        emit ChannelOpened(participantA, participantB);
    }

    function updateBalance(uint256 _balanceA, uint256 _balanceB) external onlyParticipants {
        require(isOpen, "Channel is closed");
        balanceA = _balanceA;
        balanceB = _balanceB;
        emit BalanceUpdated(msg.sender, msg.sender == participantA ? balanceA : balanceB);
    }

    function closeChannel() external onlyParticipants {
        isOpen = false;
        emit ChannelClosed(msg.sender);
    }

    function settle() external {
        require(!isOpen, "Channel is still open");
        payable(participantA).transfer(balanceA);
        payable(participantB).transfer(balanceB);
    }

    receive() external payable {}
}
