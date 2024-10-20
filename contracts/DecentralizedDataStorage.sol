pragma solidity ^0.8.0;

contract DecentralizedDataStorage {
    // Define the mapping of user addresses to their data
    mapping(address => string[]) private userData;

    // Event emitted when data is stored
    event DataStored(address indexed user, string dataHash);

    // Function to store data
    function storeData(string memory _dataHash) public {
        userData[msg.sender].push(_dataHash);
        emit DataStored(msg.sender, _dataHash);
    }

    // Function to retrieve data for a user
    function retrieveData() public view returns (string[] memory) {
        return userData[msg.sender];
    }

    // Function to get the number of data entries for a user
    function getDataCount() public view returns (uint) {
        return userData[msg.sender].length;
    }
}
