pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PIToken {
    using SafeMath for uint256;
    using SafeERC20 for address;

    // Mapping of PI token balances
    mapping (address => uint256) public balances;

    // Mapping of PI token allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Total supply of PI tokens
    uint256 public totalSupply;

    // Name of the PI token
    string public name;

    // Symbol of the PI token
    string public symbol;

    // Decimals of the PI token
    uint8 public decimals;

    // Event emitted when PI tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when PI token allowance is set
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Initializes the PI token contract
     * @param _name Name of the PI token
     * @param _symbol Symbol of the PI token
     * @param _decimals Decimals of the PI token
     * @param _initialSupply Initial supply of PI tokens
     */
    constructor(string memory _name, string memory _symbol, uint8 _decimals, uint256 _initialSupply) public {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        totalSupply = _initialSupply;
        balances[msg.sender] = _initialSupply;
    }

    /**
     * @dev Transfers PI tokens from one address to another
     * @param _from Address to transfer from
     * @param _to Address to transfer to
     * @param _value Amount of PI tokens to transfer
     */
    function transfer(address _from, address _to, uint256 _value) public {
        require(_from != address(0), "Cannot transfer from zero address");
        require(_to != address(0), "Cannot transfer to zero address");
        require(balances[_from] >= _value, "Insufficient balance");

        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);

        emit Transfer(_from, _to, _value);
    }

    /**
     * @dev Approves an address to spend PI tokens on behalf of another address
     * @param _owner Address that owns the PI tokens
     * @param _spender Address that is approved to spend PI tokens
     * @param _value Amount of PI tokens that can be spent
     */
    function approve(address _owner, address _spender, uint256 _value) public {
        require(_owner != address(0), "Cannot approve from zero address");
        require(_spender != address(0), "Cannot approve to zero address");

        allowances[_owner][_spender] = _value;

        emit Approval(_owner, _spender, _value);
    }

    /**
     * @dev Transfers PI tokens from one address to another using an approved allowance
     * @param _from Address to transfer from
     * @param _to Address to transfer to
     * @param _value Amount of PI tokens to transfer
     */
    function transferFrom(address _from, address _to, uint256 _value) public {
        require(_from != address(0), "Cannot transfer from zero address");
        require(_to != address(0), "Cannot transfer to zero address");
        require(allowances[_from][msg.sender] >= _value, "Insufficient allowance");
        require(balances[_from] >= _value, "Insufficient balance");

        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        allowances[_from][msg.sender] = allowances[_from][msg.sender].sub(_value);

        emit Transfer(_from, _to, _value);
    }

    /**
     * @dev Returns the balance of PI tokens for a given address
     * @param _address Address to retrieve balance for
     */
    function balanceOf(address _address) public view returns (uint256) {
        return balances[_address];
    }

    /**
     * @dev Returns the allowance of PI tokens for a given address and spender
     * @param _owner Address that owns the PI tokens
     * @param _spender Address that is approved to spend PI tokens
     */
    function allowance(address _owner, address _spender) public view returns (uint256) {
        return allowances[_owner][_spender];
    }
}
