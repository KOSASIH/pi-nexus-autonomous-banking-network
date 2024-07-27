pragma solidity ^0.8.0;

interface IKosasihUniversalisBridge {
    function transferTokens(address _token, address _from, address _to, uint256 _amount) external;

    function executeCrossChainTransaction(bytes _transactionData) external;
}
