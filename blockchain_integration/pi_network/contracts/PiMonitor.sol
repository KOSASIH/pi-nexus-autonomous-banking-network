pragma solidity ^0.8.0;

contract PiMonitor {
    mapping (address => uint256) public transactionCounts;
    mapping (address => uint256) public errorCounts;

    function trackTransaction(address _address) public {
        transactionCounts[_address]++;
    }

    function trackError(address _address) public {
        errorCounts[_address]++;
    }

    function getTransactionCount(address _address) public view returns (uint256) {
        return transactionCounts[_address];
    }

    function getErrorCount(address _address) public view returns (uint256) {
        return errorCounts[_address];
    }
}
