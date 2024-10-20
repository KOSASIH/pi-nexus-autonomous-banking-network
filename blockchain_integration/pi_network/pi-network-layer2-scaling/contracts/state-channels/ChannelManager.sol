// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./StateChannel.sol";

contract ChannelManager {
    mapping(address => mapping(address => StateChannel)) public channels;
    event ChannelCreated(address indexed participantA, address indexed participantB, address channelAddress);

    function createChannel(address _participantB) external {
        require(_participantB != msg.sender, "Cannot create channel with self");
        StateChannel channel = new StateChannel(msg.sender, _participantB, address(this));
        channels[msg.sender][_participantB] = channel;
        emit ChannelCreated(msg.sender, _participantB, address(channel));
    }

    function getChannel(address _participantA, address _participantB) external view returns (StateChannel) {
        return channels[_participantA][_participantB];
    }
}
