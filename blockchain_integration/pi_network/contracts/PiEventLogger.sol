pragma solidity ^0.8.0;

contract PiEventLogger {
    event LogEvent(string _event, address _address, uint256 _value);

    function logEvent(string memory _event, address _address, uint256 _value) public {
        emit LogEvent(_event, _address, _value);
    }
}
