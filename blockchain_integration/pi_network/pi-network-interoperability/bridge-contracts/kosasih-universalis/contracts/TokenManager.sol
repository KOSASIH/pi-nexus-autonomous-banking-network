pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract TokenManager {
    mapping(address => mapping(address => uint256)) public tokenBalances;

    constructor() public {
        // Initialize the token manager
    }

    function depositToken(address _token, address _user, uint256 _amount) public {
        // Deposit tokens into the token manager
        tokenBalances[_token][_user] += _amount;
    }

    function withdrawToken(address _token, address _user, uint256 _amount) public {
        // Withdraw tokens from the token manager
        require(tokenBalances[_token][_user] >= _amount, "Insufficient token balance");
        tokenBalances[_token][_user] -= _amount;
    }

    function getTokenBalance(address _token, address _user) public view returns (uint256) {
        // Retrieve a user's token balance
        return tokenBalances[_token][_user];
    }
}
