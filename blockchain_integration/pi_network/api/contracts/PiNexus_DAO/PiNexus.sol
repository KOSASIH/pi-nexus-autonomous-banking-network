// PiNexus.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/AccessControl.sol";

contract PiNexus {
    //...
    function createProposal(string memory description, address[] memory targets, uint256[] memory values, bytes[] memory data) public onlyMember {
        //...
    }

    function vote(uint256 proposalId, bool support) public onlyMember {
        //...
    }

    function executeProposal(uint256 proposalId) public onlyExecutor {
        //...
    }

    function addMember(address member, uint256 role) public onlyAdmin {
        //...
    }

    //...
}
