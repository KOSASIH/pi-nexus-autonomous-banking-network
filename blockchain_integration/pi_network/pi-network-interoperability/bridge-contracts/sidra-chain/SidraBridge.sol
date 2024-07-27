pragma solidity ^0.8.0;

import "./Pausable.sol";
import "./Owner.sol";
import "./WalletAccessControl.sol";
import "./SidraToken.sol";

contract SidraBridge is Pausable, Owner, WalletAccessControl {
    SidraToken public sidraToken;
    address public foreignChainAddress;
    uint256 public bridgeFee;

    event BridgeTransfer(address indexed from, address indexed to, uint256 value);
    event BridgeReceived(address indexed from, address indexed to, uint256 value);

    constructor() public {
        sidraToken = SidraToken(address(new SidraToken()));
        foreignChainAddress = address(0);
        bridgeFee = 0.1 * (10**18); // 0.1 ST
    }

    function setForeignChainAddress(address _foreignChainAddress) public onlyOwner {
        foreignChainAddress = _foreignChainAddress;
    }

    function setBridgeFee(uint256 _bridgeFee) public onlyOwner {
        bridgeFee = _bridgeFee;
    }

    function transferToForeignChain(address _to, uint256 _value) public whenNotPaused {
        require(isWalletAllowed(msg.sender), "Wallet not allowed");
        require(_value > 0, "Invalid value");
        require(sidraToken.balanceOf(msg.sender) >= _value, "Insufficient balance");
        sidraToken.transferFrom(msg.sender, address(this), _value);
        emit BridgeTransfer(msg.sender, _to, _value);
    }

    function receiveFromForeignChain(address _from, address _to, uint256 _value) public whenNotPaused {
        require(foreignChainAddress == msg.sender, "Invalid foreign chain address");
        require(_value > 0, "Invalid value");
        sidraToken.transfer(_to, _value);
        emit BridgeReceived(_from, _to, _value);
    }
}
