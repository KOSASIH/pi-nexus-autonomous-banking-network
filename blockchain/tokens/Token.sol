// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/security/Pausable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/Mintable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/Burnable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/IERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract Token is Pausable, Mintable, Burnable, IERC20 {
    using SafeERC20 for address;
    using SafeMath for uint256;
    using Address for address;

    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Mapping of token allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Token metadata
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Pause(address indexed account);
    event Unpause(address indexed account);
    event Mint(address indexed account, uint256 value);
    event Burn(address indexed account, uint256 value);

    // Constructor
    constructor(string memory _name, string memory _symbol, uint8 _decimals, uint256 _totalSupply) public {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        totalSupply = _totalSupply;
        _mint(msg.sender, _totalSupply);
    }

    // Transfer tokens
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(!paused(), "Token is paused");
        require(_value <= balances[msg.sender], "Insufficient balance");
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Approve token spending
    function approve(address _spender, uint256 _value) public returns (bool) {
        require(!paused(), "Token is paused");
        allowances[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    // Transfer tokens from another address
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(!paused(), "Token is paused");
        require(_value <= balances[_from], "Insufficient balance");
        require(_value<= allowances[_from][msg.sender], "Insufficient allowance");
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        allowances[_from][msg.sender] = allowances[_from][msg.sender].sub(_value);
        emit Transfer(_from, _to, _value);
        return true;
    }

    // Increase token allowance
    function increaseAllowance(address _spender, uint256 _addedValue) public returns (bool) {
        require(!paused(), "Token is paused");
        allowances[msg.sender][_spender] = allowances[msg.sender][_spender].add(_addedValue);
        emit Approval(msg.sender, _spender, allowances[msg.sender][_spender]);
        return true;
    }

    // Decrease token allowance
    function decreaseAllowance(address _spender, uint256 _subtractedValue) public returns (bool) {
        require(!paused(), "Token is paused");
        allowances[msg.sender][_spender] = allowances[msg.sender][_spender].sub(_subtractedValue);
        emit Approval(msg.sender, _spender, allowances[msg.sender][_spender]);
        return true;
    }

    // Burn tokens
    function burn(uint256 _value) public {
        require(!paused(), "Token is paused");
        require(_value <= balances[msg.sender], "Insufficient balance");
        balances[msg.sender] = balances[msg.sender].sub(_value);
        totalSupply = totalSupply.sub(_value);
        emit Burn(msg.sender, _value);
    }

    // Mint tokens
    function mint(address _to, uint256 _value) public {
        require(!paused(), "Token is paused");
        totalSupply = totalSupply.add(_value);
        balances[_to] = balances[_to].add(_value);
        emit Mint(_to, _value);
    }

    // Transfer tokens with additional data
    function transferWithData(address _to, uint256 _value, bytes memory _data) public returns (bool) {
        require(!paused(), "Token is paused");
        require(_value <= balances[msg.sender], "Insufficient balance");
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        // Add custom logic for handling data here
        return true;
    }

    // Transfer multiple tokens at once
    function batchTransfer(address[] memory _receivers, uint256[] memory _values) public returns (bool) {
        require(!paused(), "Token is paused");
        for (uint256 i = 0; i < _receivers.length; i++) {
            require(_values[i] <= balances[msg.sender], "Insufficient balance");
            balances[msg.sender] = balances[msg.sender].sub(_values[i]);
            balances[_receivers[i]] = balances[_receivers[i]].add(_values[i]);
            emit Transfer(msg.sender, _receivers[i], _values[i]);
        }
        return true;
    }

    // Pause the token
    function pause() public onlyOwner {
        _pause();
        emit Pause(msg.sender);
    }

    // Unpause the token
    function unpause() public onlyOwner {
        _unpause();
        emit Unpause(msg.sender);
    }
}
