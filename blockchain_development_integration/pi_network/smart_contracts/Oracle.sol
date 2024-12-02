// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Oracle {
    mapping(bytes32 => uint256) public data;

    function updateData(bytes32 key, uint256 value) external {
        // In a real-world scenario, this function would be called by a trusted data provider
        data[key] = value;
    }

    function getData(bytes32 key) external view returns (uint256) {
        return data[key];
    }
}
