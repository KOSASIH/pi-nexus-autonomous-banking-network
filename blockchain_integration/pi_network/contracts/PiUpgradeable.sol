pragma solidity ^0.8.0;

contract PiUpgradeable {
    address public implementation;

    constructor(address _implementation) public {
        implementation = _implementation;
    }

    function upgrade(address _newImplementation) public {
        require(msg.sender == implementation, "Only the implementation can upgrade");
        implementation = _newImplementation;
    }

    function delegatecall(address _target, bytes memory _data) public {
        require(implementation == _target, "Only the implementation can be called");
        (bool success, bytes memory returnData) = _target.delegatecall(_data);
        require(success, "Delegatecall failed");
        assembly {
            return(add(returnData, 0x20), mload(returnData))
        }
    }
}
