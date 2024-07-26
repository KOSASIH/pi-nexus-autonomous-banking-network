pragma solidity ^0.8.0;

contract AtomicSwap {
    // Mapping of swap IDs to their respective swap details
    mapping (bytes32 => SwapDetails) public swaps;

    // Event emitted when a swap is executed
    event SwapExecuted(bytes32 indexed swapId, address indexed sender, address indexed recipient, uint256 amount);

    // Function to execute a swap
    function executeSwap(bytes32 swapId, address sender, address recipient, uint256 amount) public {
        // Check if the swap exists
        require(swaps[swapId].sender != address(0), "Swap does not exist");

        // Transfer the tokens
        swaps[swapId].token.safeTransfer(recipient, amount);

        // Emit the SwapExecuted event
        emit SwapExecuted(swapId, sender, recipient, amount);
    }
}
