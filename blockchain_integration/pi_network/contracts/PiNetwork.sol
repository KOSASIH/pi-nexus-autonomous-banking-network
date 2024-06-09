pragma solidity ^0.8.0;

import "./PiToken.sol";
import "./PiNode.sol";
import "./PiWallet.sol";
import "./PiBanking.sol";
import "./PiGovernance.sol";

contract PiNetwork {
    address public owner;
    PiToken public piToken;
    PiNode public piNode;
    PiWallet public piWallet;
    PiBanking public piBanking;
    PiGovernance public piGovernance;

    constructor() public {
        owner = msg.sender;
        piToken = new PiToken();
        piNode = new PiNode();
        piWallet = new PiWallet();
        piBanking = new PiBanking();
        piGovernance = new PiGovernance();
    }

    function transferPi(address _to, uint256 _value) public {
        require(msg.sender == owner, "Only the owner can transfer PI");
        piToken.transfer(_to, _value);
    }

    function getNodeAddress() public view returns (address) {
        return piNode.getAddress();
    }

    function getWalletAddress() public view returns (address) {
        return piWallet.getAddress();
    }

    function getBankingAddress() public view returns (address) {
        return piBanking.getAddress();
    }

    function getGovernanceAddress() public view returns (address) {
        return piGovernance.getAddress();
    }
}
