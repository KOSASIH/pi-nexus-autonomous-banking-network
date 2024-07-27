pragma solidity ^0.8.0;

interface IKosasihUniversalisToken is ERC20 {
    function name() external view returns (string);

    function symbol() external view returns (string);

    function totalSupply() external view returns (uint256);
}
