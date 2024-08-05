pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract WalletContract {
    using SafeMath for uint256;
    using SafeERC20 for ERC20;

    // Mapping of user addresses to their respective wallet balances
    mapping (address => mapping (address => uint256)) public balances;

    // Mapping of user addresses to their respective wallet allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Event emitted when a user deposits tokens into their wallet
    event Deposit(address indexed user, address indexed token, uint256 amount);

    // Event emitted when a user withdraws tokens from their wallet
    event Withdrawal(address indexed user, address indexed token, uint256 amount);

    // Event emitted when a user transfers tokens to another user
    event Transfer(address indexed from, address indexed to, address indexed token, uint256 amount);

    // Function to deposit tokens into a user's wallet
    function deposit(address _token, uint256 _amount) public {
        ERC20 token = ERC20(_token);
        token.safeTransferFrom(msg.sender, address(this), _amount);
        balances[msg.sender][_token] = balances[msg.sender][_token].add(_amount);
        emit Deposit(msg.sender, _token, _amount);
    }

    // Function to withdraw tokens from a user's wallet
    function withdraw(address _token, uint256 _amount) public {
        require(balances[msg.sender][_token] >= _amount, "Insufficient balance");
        ERC20 token = ERC20(_token);
        token.safeTransfer(msg.sender, _amount);
        balances[msg.sender][_token] = balances[msg.sender][_token].sub(_amount);
        emit Withdrawal(msg.sender, _token, _amount);
    }

    // Function to transfer tokens from one user to another
    function transfer(address _to, address _token, uint256 _amount) public {
        require(balances[msg.sender][_token] >= _amount, "Insufficient balance");
        balances[msg.sender][_token] = balances[msg.sender][_token].sub(_amount);
        balances[_to][_token] = balances[_to][_token].add(_amount);
        emit Transfer(msg.sender, _to, _token, _amount);
    }

    // Function to approve a user to spend tokens on behalf of another user
    function approve(address _spender, address _token, uint256 _amount) public {
        allowances[msg.sender][_spender] = _amount;
    }

    // Function to check a user's wallet balance
    function balanceOf(address _user, address _token) public view returns (uint256) {
        return balances[_user][_token];
    }

    // Function to check a user's wallet allowance
    function allowance(address _user, address _spender, address _token) public view returns (uint256) {
        return allowances[_user][_spender];
    }
}
