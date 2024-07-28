pragma solidity ^0.8.0;

import {DeFiIntegration} from "./DeFiIntegration.sol";
import {ERC20} from "./ERC20.sol";

contract BorrowingProtocol {
    // Mapping of borrowers to their corresponding borrowing balances
    mapping (address => uint256) public borrowingBalances;

    // Event emitted when a borrower borrows Pi Coin from the borrowing protocol
    event BorrowerBorrow(address indexed borrower, uint256 amount);

    // Event emitted when a borrower repays a loan on the borrowing protocol
    event BorrowerRepay(address indexed borrower, uint256 amount);

    // Function to borrow Pi Coin from the borrowing protocol
    function borrow(uint256 _amount) public {
        // Check if borrowing protocol has sufficient liquidity
        require(DeFiIntegration.defiBalances[msg.sender] >= _amount, "Insufficient liquidity");

        // Transfer Pi Coin from borrowing protocol to borrower
        DeFiIntegration.transfer(msg.sender, _amount);

        // Update borrower's borrowing balance
        borrowingBalances[msg.sender] += _amount;

        // Emit borrower borrow event
        emit BorrowerBorrow(msg.sender, _amount);
    }

    // Function to repay a loan on the borrowing protocol
    function repay(uint256 _amount) public {
        // Check if borrower has sufficient DeFi balance
        require(DeFiIntegration.defiBalances[msg.sender] >= _amount, "Insufficient DeFi balance");

        // Transfer Pi Coin from borrower to borrowing protocol
        DeFiIntegration.transferFrom(msg.sender, address(this), _amount);

        // Update borrower's borrowing balance
        borrowingBalances[msg.sender] -= _amount;

        // Emit borrower repay event
        emit BorrowerRepay(msg.sender, _amount);
    }
}
