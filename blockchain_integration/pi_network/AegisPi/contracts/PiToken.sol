pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract PiToken is ERC20, ERC20Burnable, ERC20Pausable, AccessControl {
    // Mapping of addresses to their respective balances
    mapping(address => uint256) public balances;

    // Mapping of addresses to their respective allowances
    mapping(address => mapping(address => uint256)) public allowances;

    // Total supply of Pi Tokens
    uint256 public totalSupply;

    // Name of the token
    string public name;

    // Symbol of the token
    string public symbol;

    // Decimals of the token
    uint8 public decimals;

    // Role-based access control
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    // Events
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    event Pause();
    event Unpause();

    // Constructor
    constructor() public ERC20("Pi Token", "PI") {
        // Initialize total supply
        totalSupply = 1000000000;

        // Initialize name, symbol, and decimals
        name = "Pi Token";
        symbol = "PI";
        decimals = 18;

        // Initialize role-based access control
        _setupRole(MINTER_ROLE, msg.sender);
        _setupRole(BURNER_ROLE, msg.sender);
        _setupRole(PAUSER_ROLE, msg.sender);
    }

    // Function to mint new tokens
    function mint(address to, uint256 amount) public {
        require(hasRole(MINTER_ROLE, msg.sender), "PiToken: only minter can mint");
        require(amount > 0, "PiToken: amount must be greater than 0");

        // Update total supply
        totalSupply += amount;

        // Update balances
        balances[to] += amount;

        // Emit event
        emit Mint(to, amount);
    }

    // Function to burn tokens
    function burn(uint256 amount) public {
        require(hasRole(BURNER_ROLE, msg.sender), "PiToken: only burner can burn");
        require(amount > 0, "PiToken: amount must be greater than 0");
        require(balances[msg.sender] >= amount, "PiToken: insufficient balance");

        // Update total supply
        totalSupply -= amount;

        // Update balances
        balances[msg.sender] -= amount;

        // Emit event
        emit Burn(msg.sender, amount);
    }

    // Function to pause token transfers
    function pause() public {
        require(hasRole(PAUSER_ROLE, msg.sender), "PiToken: only pauser can pause");
        _pause();
        emit Pause();
    }

    // Function to unpause token transfers
    function unpause() public {
        require(hasRole(PAUSER_ROLE, msg.sender), "PiToken: only pauser can unpause");
        _unpause();
        emit Unpause();
    }
}
