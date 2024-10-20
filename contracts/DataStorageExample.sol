pragma solidity ^0.8.0;

contract DataStorageExample {
    DecentralizedDataStorage public dataStorage;

    constructor(address _dataStorageAddress) {
        dataStorage = DecentralizedDataStorage(_dataStorageAddress);
    }

    // Function to store data
    function storeUser Data(string memory _dataHash) public {
        dataStorage.storeData(_dataHash);
    }

    // Function to retrieve user data
    function retrieveUser Data() public view returns (string[] memory) {
        return dataStorage.retrieveData();
    }

    // Function to get the count of user data entries
    function getUser DataCount() public view returns (uint) {
        return dataStorage.getDataCount();
    }
}
