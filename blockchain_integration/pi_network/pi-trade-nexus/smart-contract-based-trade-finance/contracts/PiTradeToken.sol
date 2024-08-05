pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiTradeToken is ERC20 {
    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Event emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when tokens are approved for transfer
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Total supply of PiTradeToken
    uint256 public totalSupply;

    // Name of the token
    string public name;

    // Symbol of the token
    string public symbol;

    // Decimals of the token
    uint8 public decimals;

    // Constructor to initialize the token
    constructor() public {
        totalSupply = 100000000 * (10 ** decimals);
        name = "PiTradeToken";
        symbol = "PTT";
        decimals = 18;

        // Allocate initial supply of tokens to the contract creator
        balances[msg.sender] = totalSupply;
    }

    // Function to transfer tokens
    function transfer(address to, uint256 value) public returns (bool) {
        require(to!= address(0), "Recipient address cannot be zero");
        require(value > 0, "Value must be greater than zero");

        balances[msg.sender] -= value;
        balances[to] += value;

        emit Transfer(msg.sender, to, value);

        return true;
    }

    // Function to approve tokens for transfer
    function approve(address spender, uint256 value) public returns (bool) {
        require(spender!= address(0), "Spender address cannot be zero");
        require(value > 0, "Value must be greater than zero");

        allowances[msg.sender][spender] = value;

                emit Approval(msg.sender, spender, value);

        return true;
    }

    // Function to transfer tokens from one address to another
    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(from!= address(0), "From address cannot be zero");
        require(to!= address(0), "To address cannot be zero");
        require(value > 0, "Value must be greater than zero");

        balances[from] -= value;
        balances[to] += value;

        emit Transfer(from, to, value);

        return true;
    }

    // Function to get the balance of a specific address
    function balanceOf(address owner) public view returns (uint256) {
        return balances[owner];
    }

    // Function to get the allowance of a specific address for a spender
    function allowance(address owner, address spender) public view returns (uint256) {
        return allowances[owner][spender];
    }
}
