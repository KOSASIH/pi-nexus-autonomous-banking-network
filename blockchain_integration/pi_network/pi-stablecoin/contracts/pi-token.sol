pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "./PiOracle.sol";

contract PiToken is ERC20, Ownable {
    // The Pi Oracle contract
    PiOracle public piOracle;

    // The total supply of Pi tokens
    uint256 public constant TOTAL_SUPPLY = 1000000000 * (10**18); // 1 billion Pi tokens

    // The decimal places for the Pi token
    uint256 public constant DECIMALS = 18;

    // Event emitted when Pi tokens are minted
    event Mint(address indexed recipient, uint256 amount);

    // Event emitted when Pi tokens are burned
    event Burn(address indexed recipient, uint256 amount);

    // Event emitted when the Pi Oracle contract is updated
    event UpdatePiOracle(address indexed newPiOracle);

    /**
     * @dev Initializes the PiToken contract with the Pi Oracle contract.
     * @param _piOracle The address of the Pi Oracle contract.
     */
    constructor(address _piOracle) public {
        piOracle = PiOracle(_piOracle);
        _mint(msg.sender, TOTAL_SUPPLY);
    }

    /**
     * @dev Mints Pi tokens to a recipient.
     * @param _recipient The address of the recipient.
     * @param _amount The amount of Pi tokens to mint.
     */
    function mint(address _recipient, uint256 _amount) public onlyOwner {
        require(_amount > 0, "Amount must be greater than 0");
        _mint(_recipient, _amount);
        emit Mint(_recipient, _amount);
    }

    /**
     * @dev Burns Pi tokens from a recipient.
     * @param _recipient The address of the recipient.
     * @param _amount The amount of Pi tokens to burn.
     */
    function burn(address _recipient, uint256 _amount) public onlyOwner {
        require(_amount > 0, "Amount must be greater than 0");
        _burn(_recipient, _amount);
        emit Burn(_recipient, _amount);
    }

    /**
     * @dev Updates the Pi Oracle contract.
     * @param _newPiOracle The address of the new Pi Oracle contract.
     */
    function updatePiOracle(address _newPiOracle) public onlyOwner {
        piOracle = PiOracle(_newPiOracle);
        emit UpdatePiOracle(_newPiOracle);
    }

    /**
     * @dev Returns the current Pi price from the Pi Oracle contract.
     * @return The current Pi price in wei.
     */
    function getPiPrice() public view returns (uint256) {
        return piOracle.getPiPrice();
    }
}
