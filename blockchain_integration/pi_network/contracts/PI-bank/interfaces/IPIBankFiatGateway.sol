pragma solidity ^0.8.4;

interface IPIBankFiatGateway {
    function deposit(uint256 fiatAmount) external;
    function withdraw(uint256 fiatAmount) external;
    function getFiatBalance(address user) external view returns (uint256);
}
