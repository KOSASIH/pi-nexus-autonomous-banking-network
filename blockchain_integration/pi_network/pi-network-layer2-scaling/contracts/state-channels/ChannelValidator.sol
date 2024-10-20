// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./StateChannel.sol";

contract ChannelValidator {
    event ChannelValidated(address indexed channel, address indexed participant);

    function validateChannel(StateChannel _channel) external {
        require(!_channel.isOpen(), "Channel is still open");
        emit ChannelValidated(address(_channel), msg.sender);
    }

    function validateFinalBalances(StateChannel _channel, uint256 _finalBalanceA, uint256 _finalBalanceB) external {
        require(!_channel.isOpen(), "Channel is still open");
        require(_finalBalanceA + _finalBalanceB == address(_channel).balance, "Invalid final balances");
        emit ChannelValidated(address(_channel), msg.sender);
    }
}
