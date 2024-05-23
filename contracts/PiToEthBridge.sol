pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract PiToEthBridge {

    // The PI Network contract address
    address public piNetworkContract;

    // The Ethereum contract address
    address public ethereumContract;

    // The PI-to-Ethereum conversion rate
    uint public conversionRate;

    // The function to initialize the contract
    constructor(address _piNetworkContract, address _ethereumContract, uint _conversionRate) {
        piNetworkContract = _piNetworkContract;
        ethereumContract = _ethereumContract;
        conversionRate = _conversionRate;
    }

    // The function to convert PI tokens to Ethereum tokens
    function convertPiToEthereum(uint _amount) external {
        IERC20 piToken = IERC20(piNetworkContract);
        require(piToken.balanceOf(msg.sender) >= _amount, "Insufficient PI token balance");

        uint ethereumAmount = (_amount * conversionRate) / 100;

        piToken.transferFrom(msg.sender, address(this), _amount);

        IERC20 ethereumToken = IERC20(ethereumContract);
        ethereumToken.transfer(msg.sender, ethereumAmount);
    }

    // The function to convert Ethereum tokens to PI tokens
    function convertEthereumToPi(uint _amount) external {
        IERC20 ethereumToken = IERC20(ethereumContract);
        require(ethereumToken.balanceOf(msg.sender) >= _amount, "Insufficient Ethereum token balance");

        uint piAmount = (_amount * 100) / conversionRate;

        ethereumToken.transferFrom(msg.sender, address(this), _amount);

        IERC20 piToken = IERC20(piNetworkContract);
        piToken.transfer(msg.sender, piAmount);
    }

}
