pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiTradeToken {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Mapping of user balances
    mapping (address => uint256) public balances;

    // Mapping of user allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Total supply of tokens
    uint256 public totalSupply;

    // Token name
    string public name;

    // Token symbol
    string public symbol;

    // Token decimals
    uint8 public decimals;

    // Event emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when an approval is made
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Initializes the contract with the token name, symbol, and decimals
     */
    constructor() public {
        name = "Pi Trade Token";
        symbol = "PTT";
        decimals = 18;
        totalSupply = 100000000 * (10 ** decimals);
        balances[msg.sender] = totalSupply;
    }

    /**
     * @dev Transfers tokens from one address to another
     * @param from The address to transfer from
     * @param to The address to transfer to
     * @param value The amount of tokens to transfer
     */
    function transfer(address from, address to, uint256 value) public {
        require(from!= address(0), "ERC20: transfer from the zero address");
        require(to!= address(0), "ERC20: transfer to the zero address");
        require(value <= balances[from], "ERC20: transfer amount exceeds balance");

        balances[from] = balances[from].sub(value);
        balances[to] = balances[to].add(value);
        emit Transfer(from, to, value);
    }

    /**
     * @dev Approves an address to spend tokens on behalf of another address
     * @param owner The address that owns the tokens
     * @param spender The address that is approved to spend tokens
     * @param value The amount of tokens that can be spent
     */
    function approve(address owner, address spender, uint256 value) public {
        require(owner!= address(0), "ERC20: approve from the zero address");
        require(spender!= address(0), "ERC20: approve to the zero address");

        allowances[owner][spender] = value;
        emit Approval(owner, spender, value);
    }

    /**
     * @dev Transfers tokens from one address to another using an allowance
     * @param from The address to transfer from
     * @param to The address to transfer to
     * @param value The amount of tokens to transfer
     */
    function transferFrom(address from, address to, uint256 value) public {
        require(from!= address(0), "ERC20: transfer from the zero address");
        require(to!= address(0), "ERC20: transfer to the zero address");
        require(value <= balances[from], "ERC20: transfer amount exceeds balance");
        require(value <= allowances[from][msg.sender], "ERC20: transfer amount exceeds allowance");

        balances[from] = balances[from].sub(value);
        balances[to] = balances[to].add(value);
        allowances[from][msg.sender] = allowances[from][msg.sender].sub(value);
        emit Transfer(from, to, value);
    }
}
