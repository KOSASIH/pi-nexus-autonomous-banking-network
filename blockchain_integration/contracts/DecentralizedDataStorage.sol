pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedDataStorage {
    // Mapping of data hashes to data contents
    mapping (bytes32 => bytes) public dataContents;

    // Event emitted when new data is stored
    event DataStored(bytes32 dataHash, bytes dataContent);

    // Function to store new data
    function storeData(bytes memory _dataContent) public {
        // Calculate data hash
        bytes32 dataHash = keccak256(_dataContent);

        // Store data content
        dataContents[dataHash] = _dataContent;

        // Emit data stored event
        emit DataStored(dataHash, _dataContent);
    }

    // Function to retrieve data
    function retrieveData(bytes32 _dataHash) public view returns (bytes memory) {
        return dataContents[_dataHash];
    }
}
