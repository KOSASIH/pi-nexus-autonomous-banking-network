interface SmartContractInterface {
    function deployContract(bytes memory bytecode) external returns (address);
    function executeContract(address contractAddress, bytes memory input) external returns (bytes memory);
    function getContractBalance(address contractAddress) external view returns (uint256);
}
