pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/blob/master/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/blob/master/contracts/lifecycle/Pausable.sol";

contract PiNetworkContract is Ownable, Pausable {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function addNode(address _node) public {
        require(msg.sender == owner, "Only the owner can add nodes");
        nodes[_node] = true;
    }

    function removeNode(address _node) public {
        require(msg.sender == owner, "Only the owner can remove nodes");
        delete nodes[_node];
    }

    function getBalance(address _address) public view returns (uint256) {
        return balances[_address];
    }

    function setBalance(address _address, uint256 _balance) public {
        require(msg.sender == owner, "Only the owner can set balances");
        balances[_address] = _balance;
    }
}
