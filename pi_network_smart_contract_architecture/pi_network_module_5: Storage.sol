pragma solidity ^0.8.0;

import "./AccessControl.sol";

contract Storage {
    using AccessControl for address;

    // Mapping of storage data
    mapping (string => bytes) public data;

    // Event emitted when new storage data is updated
    event NewData(string indexed key, bytes value);

    // Function to update storage data
    function updateData(string memory _key, bytes memory _value) public onlyAdmin {
        data[_key] = _value;
        emit NewData(_key, _value);
    }

    // Function to get storage data
    function getData(string memory _key) public view returns (bytes memory) {
        return data[_key];
    }
}
