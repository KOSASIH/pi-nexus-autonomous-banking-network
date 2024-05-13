pragma solidity ^0.8.0;

contract StorageContract {
    struct File {
        string name;
        string hash;
        address owner;
    }

    mapping(address => File[]) private files;

    event FileStored(address indexed owner, uint256 indexed fileId, string name, string hash);

    function storeFile(string memory _name, string memory _hash) public {
        File memory newFile;
        newFile.name = _name;
        newFile.hash = _hash;
        newFile.owner = msg.sender;

        uint256 fileId = files[msg.sender].length;
        files[msg.sender].push(newFile);

        emit FileStored(msg.sender, fileId, _name, _hash);
    }

    function getFile(address _owner, uint256 _fileId) public view returns (string memory, string memory) {
        File storage file = files[_owner][_fileId];
        return (file.name, file.hash);
    }

    function deleteFile(address _owner, uint256 _fileId) public {
        delete files[_owner][_fileId];
    }
}
