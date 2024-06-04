pragma solidity ^0.8.0;

import "https://github.com/uport/uport-identity/blob/master/contracts/Identity.sol";

contract PiNetworkIdentity is Identity {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function addIdentity(address _identity) public {
        require(msg.sender == owner, "Only the owner can add identities");
        identities[_identity] = true;
    }

    function removeIdentity(address _identity) public {
        require(msg.sender == owner, "Only the owner can remove identities");
        delete identities[_identity];
    }

    function getBalance(address _address) public view returns (uint256) {
        return balances[_address];
    }

    function setBalance(address _address, uint256 _balance) public {
        require(msg.sender == owner, "Only the owner can set balances");
        balances[_address] = _balance;
    }
}
