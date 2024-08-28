pragma solidity ^0.8.0;

import "./Governance.sol";

contract CommunityEngagement {
    using Governance for address;

    // Mapping of community members
    mapping (address => CommunityMember) public communityMembers;

    // Struct to represent a community member
    struct CommunityMember {
        address member;
        uint256 reputation;
        uint256 contribution;
    }

    // Event emitted when a new community member joins
    event NewCommunityMember(address indexed member);

    // Event emitted when a community member contributes to the network
    event ContributeToNetwork(address indexed member, uint256 contribution);

    // Function to join the community
    function joinCommunity() public {
        CommunityMember storage member = communityMembers[msg.sender];
        member.member = msg.sender;
        member.reputation = 0;
        member.contribution = 0;
        emit NewCommunityMember(msg.sender);
    }

    // Function to contribute to the network
    function contributeToNetwork(uint256 _contribution) public {
        CommunityMember storage member = communityMembers[msg.sender];
        member.contribution += _contribution;
        emit ContributeToNetwork(msg.sender, _contribution);
    }

    // Function to get a community member's reputation
    function getReputation(address _member) public view returns (uint256) {
        CommunityMember storage member = communityMembers[_member];
        return member.reputation;
    }

    // Function to get a community member's contribution
    function getContribution(address _member) public view returns (uint256) {
        CommunityMember storage member = communityMembers[_member];
        return member.contribution;
    }
}
