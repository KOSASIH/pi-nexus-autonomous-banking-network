pragma solidity ^0.8.0;

import "./IPIBankDataOracle.sol";

contract PIBankDataOracle is IPIBankDataOracle {
    mapping(string => bytes) public data;

    function fetchData(string calldata _key) public {
        // implement data fetching logic
    }

    function updateData(string calldata _key, bytes calldata _data) public {
        data[_key] = _data;
    }

    function getData(string calldata _key) public view returns (bytes memory) {
        return data[_key];
    }
}
