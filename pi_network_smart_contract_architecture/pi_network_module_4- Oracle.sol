pragma solidity ^0.8.0;

import "./AccessControl.sol";

contract Oracle {
    using AccessControl for address;

    // Mapping of oracle data
    mapping (string => uint256) public data;

    // Event emitted when new oracle data is updated
    event NewData(string indexed key, uint256 value);

    // Function to update oracle data
    function updateData(string memory _key, uint256 _value) public onlyAdmin {
        data[_key] = _value;
        emit NewData(_key, _value);
    }

    // Function to get oracle data
    function getData(string memory _key) public view returns (uint256) {
        return data[_key];
    }
}
