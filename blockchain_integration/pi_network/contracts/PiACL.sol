pragma solidity ^0.8.0;

contract PiACL {
    mapping (address => mapping (string => bool)) public accessControl;

    function setAccess(address _address, string memory _function, bool _allowed) public {
        accessControl[_address][_function] = _allowed;
    }

    modifier onlyAllowed(string memory _function) {
        require(accessControl[msg.sender][_function], "Access denied");
        _;
    }
}
