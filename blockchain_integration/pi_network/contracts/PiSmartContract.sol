pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiSmartContract {
    // Mapping of contract addresses to their respective code
    mapping (address => bytes) public contractCode;

    // Mapping of contract addresses to their respective storage
    mapping (address => mapping (string => uint256)) public contractStorage;

    // Event emitted when a new contract is created
    event NewContract(address indexed contractAddress, bytes code);

    // Event emitted when a contract is executed
    event ContractExecuted(address indexed contractAddress, bytes data);

    // Event emitted when a contract emits an event
    event ContractEvent(address indexed contractAddress, bytes data);

    /**
     * @dev Creates a new smart contract on the Pi Network
     * @param _code The bytecode of the contract
     */
    function createContract(bytes _code) public {
        address newContractAddress = address(new bytes(_code));
        contractCode[newContractAddress] = _code;
        emit NewContract(newContractAddress, _code);
    }

    /**
     * @dev Executes a smart contract on the Pi Network
     * @param _contractAddress The address of the contract to execute
     * @param _data The data to pass to the contract
     */
    function executeContract(address _contractAddress, bytes _data) public {
        require(contractCode[_contractAddress] != 0, "Contract not found");
        (bool success, bytes memory returnData) = _contractAddress.call(_data);
        require(success, "Contract execution failed");
        emit ContractExecuted(_contractAddress, returnData);
    }

    /**
     * @dev Stores data in a contract's storage
     * @param _contractAddress The address of the contract
     * @param _key The key to store the data under
     * @param _value The value to store
     */
    function setStorage(address _contractAddress, string _key, uint256 _value) public {
        require(contractCode[_contractAddress] != 0, "Contract not found");
        contractStorage[_contractAddress][_key] = _value;
    }

    /**
     * @dev Retrieves data from a contract's storage
     * @param _contractAddress The address of the contract
     * @param _key The key to retrieve the data for
     * @return The stored value
     */
    function getStorage(address _contractAddress, string _key) public view returns (uint256) {
        require(contractCode[_contractAddress] != 0, "Contract not found");
        return contractStorage[_contractAddress][_key];
    }

    /**
     * @dev Emits an event from a contract
     * @param _contractAddress The address of the contract
     * @param _data The data to emit
     */
    function emitEvent(address _contractAddress, bytes _data) public {
        require(contractCode[_contractAddress] != 0, "Contract not found");
        emit ContractEvent(_contractAddress, _data);
    }
}
