// File: MultiSigToken.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract MultiSigToken {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Multi-signature wallet addresses
    address[] public owners;
    uint256 public requiredSignatures;

    // Event emitted when a token is transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when a new owner is added
    event OwnerAdded(address indexed owner);

    // Event emitted when a new signature requirement is set
    event SignatureRequirementUpdated(uint256 requiredSignatures);

    /**
     * @dev Initializes the token contract with the initial supply and multi-signature wallet owners
     * @param _initialSupply The initial token supply
     * @param _owners The initial multi-signature wallet owners
     * @param _requiredSignatures The initial signature requirement
     */
    constructor(uint256 _initialSupply, address[] memory _owners, uint256 _requiredSignatures) public {
        balances[msg.sender] = _initialSupply;
        owners = _owners;
        requiredSignatures = _requiredSignatures;
    }

    /**
     * @dev Transfers tokens from one address to another
     * @param _from The address to transfer from
     * @param _to The address to transfer to
     * @param _value The amount of tokens to transfer
     */
    function transfer(address _from, address _to, uint256 _value) public {
        require(balances[_from] >= _value, "Insufficient balance");
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
    }

    /**
     * @dev Adds a new owner to the multi-signature wallet
     * @param _newOwner The new owner to add
     */
    function addOwner(address _newOwner) public {
        require(msg.sender == owners[0], "Only the primary owner can add new owners");
        owners.push(_newOwner);
        emit OwnerAdded(_newOwner);
    }

    /**
     * @dev Updates the signature requirement for the multi-signature wallet
     * @param _newRequiredSignatures The new signature requirement
     */
    function updateSignatureRequirement(uint256 _newRequiredSignatures) public {
        require(msg.sender == owners[0], "Only the primary owner can update the signature requirement");
        requiredSignatures = _newRequiredSignatures;
        emit SignatureRequirementUpdated(_newRequiredSignatures);
    }
}
