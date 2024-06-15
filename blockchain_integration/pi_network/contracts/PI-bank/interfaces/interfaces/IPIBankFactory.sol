pragma solidity ^0.8.4;

interface IPIBankFactory {
    function createPIBank() external;
    function getPIBank(address user) external view returns (IPIBank);
}
