pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract EthereumBridge {
    // Address of the CrossChainBridge contract
    address public crossChainBridgeAddress;

    // Mapping of token addresses to their respective Ethereum token addresses
    mapping (address => address) public ethereumTokenAddresses;

    // Event emitted when a cross-chain transfer is initiated
    event TransferInitiated(address indexed from, address indexed to, uint256 value, bytes data);

    // Event emitted when a cross-chain transfer is completed
    event TransferCompleted(address indexed from, address indexed to, uint256 value, bytes data);

    // Function to initiate a cross-chain transfer from Ethereum to another blockchain network
    function transferToOtherChain(address _to, uint256 _value, bytes _data) public {
        require(_to != address(0), "Invalid to address");
        require(_value > 0, "Invalid value");

        // Get the Ethereum token address
        address ethereumTokenAddress = ethereumTokenAddresses[_to];

        // Transfer the tokens from the Ethereum blockchain to the CrossChainBridge contract
        SafeERC20.safeTransfer(ethereumTokenAddress, crossChainBridgeAddress, _value);

        // Emit the TransferInitiated event
        emit TransferInitiated(address(this), _to, _value, _data);
    }

    // Function to complete a cross-chain transfer from another blockchain network to Ethereum
    function completeTransferFromOtherChain(address _from, uint256 _value, bytes _data) public {
        require(_from != address(0), "Invalid from address");
        require(_value > 0, "Invalid value");

        // Get the Ethereum token address
        address ethereumTokenAddress = ethereumTokenAddresses[_from];

        // Transfer the tokens from the CrossChainBridge contract to the Ethereum blockchain
        SafeERC20.safeTransfer(crossChainBridgeAddress, ethereumTokenAddress, _value);

        // Emit the TransferCompleted event
        emit TransferCompleted(_from, address(this), _value, _data);
    }
}
