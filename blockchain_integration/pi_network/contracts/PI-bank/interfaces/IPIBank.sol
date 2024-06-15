pragma solidity ^0.8.4;

interface IPIBank {
    function transfer(address to, uint256 value) external;
    function approve(address spender, uint256 value) external;
    function transferFrom(address from, address to, uint256 value) external;
    function getBalance(address user) external view returns (uint256);
    function getAllowance(address owner, address spender) external view returns (uint256);
}
