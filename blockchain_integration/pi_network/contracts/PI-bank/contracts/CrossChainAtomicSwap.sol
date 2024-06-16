pragma solidity ^0.8.0;

contract CrossChainAtomicSwap {
    mapping (address => mapping (address => uint256)) public swaps;

    function initiateSwap(address recipient, address token, uint256 amount) public {
        // Initiate a cross-chain atomic swap
        swaps[msg.sender][recipient] = amount;
    }

    function confirmSwap(address sender, address token, uint256 amount) public {
        // Confirm the cross-chain atomic swap
        require(swaps[sender][msg.sender] == amount, "Invalid swap amount");
        swaps[sender][msg.sender] = 0;
        // Transfer the tokens to the recipient
        //...
    }
}
