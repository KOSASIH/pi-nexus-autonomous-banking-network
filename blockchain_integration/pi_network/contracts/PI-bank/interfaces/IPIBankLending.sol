pragma solidity ^0.8.4;

interface IPIBankLending {
    function lend(uint256 loanAmount) external;
    function repay(uint256 loanAmount) external;
    function getLoan(address borrower) external view returns (uint256);
}
