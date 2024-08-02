pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "./pi-oracle.sol";

contract PiStablecoin {
    // Mapping of user balances
    mapping (address => uint256) public balances;

    // Mapping of allowed spenders
    mapping (address => mapping (address => uint256)) public allowed;

    // Total supply of stablecoins
    uint256 public totalSupply;

    // Pi Oracle contract instance
    PiOracle public piOracle;

    // Event emitted when stablecoins are minted
    event Mint(address indexed to, uint256 amount);

    // Event emitted when stablecoins are burned
    event Burn(address indexed from, uint256 amount);

    // Event emitted when stablecoins are transferred
    event Transfer(address indexed from, address indexed to, uint256 amount);

    // Event emitted when spenders are approved
    event Approval(address indexed owner, address indexed spender, uint256 amount);

    // Constructor
    constructor(address _piOracle) public {
        piOracle = PiOracle(_piOracle);
    }

    // Function to mint stablecoins
    function mint(address _to, uint256 _amount) public {
        // Get the current Pi price from the oracle
        uint256 piPrice = piOracle.getPiPrice();

        // Calculate the amount of stablecoins to mint
        uint256 stablecoinAmount = _amount * piPrice;

        // Update the total supply
        totalSupply += stablecoinAmount;

        // Update the user's balance
        balances[_to] += stablecoinAmount;

        // Emit the mint event
        emit Mint(_to, stablecoinAmount);
    }

    // Function to burn stablecoins
    function burn(address _from, uint256 _amount) public {
        // Get the current Pi price from the oracle
        uint256 piPrice = piOracle.getPiPrice();

        // Calculate the amount of stablecoins to burn
        uint256 stablecoinAmount = _amount * piPrice;

        // Update the total supply
        totalSupply -= stablecoinAmount;

        // Update the user's balance
        balances[_from] -= stablecoinAmount;

        // Emit the burn event
        emit Burn(_from, stablecoinAmount);
    }

    // Function to transfer stablecoins
    function transfer(address _to, uint256 _amount) public {
        // Update the user's balance
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;

        // Emit the transfer event
        emit Transfer(msg.sender, _to, _amount);
    }

    // Function to approve spenders
    function approve(address _spender, uint256 _amount) public {
        // Update the allowed spenders
        allowed[msg.sender][_spender] = _amount;

        // Emit the approval event
        emit Approval(msg.sender, _spender, _amount);
    }
}
