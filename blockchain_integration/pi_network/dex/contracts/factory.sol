// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";
import "https://github.com/Uniswap/uniswap-v2-core/contracts/UniswapV2Factory.sol";
import "https://github.com/pi-network/pi-network-contracts/contracts/token/ERC20.sol";

contract PiDEXFactory is Ownable, UniswapV2Factory {
    using Address for address;

    // Mapping of token pairs to their corresponding exchange contracts
    mapping(address => mapping(address => address)) public exchangeContracts;

    // Mapping of token pairs to their corresponding exchange contract creators
    mapping(address => mapping(address => address)) public exchangeContractCreators;

    // Event emitted when a new exchange contract is deployed
    event NewExchangeContract(address tokenA, address tokenB, address exchangeContract, address creator);

    // Function to deploy a new exchange contract
    function createExchangeContract(address tokenA, address tokenB) public {
        require(tokenA!= tokenB, "Cannot create exchange contract for same token");
        require(exchangeContracts[tokenA][tokenB] == address(0), "Exchange contract already exists");

        // Deploy a new exchange contract
        address exchangeContract = address(new ExchangeContract(tokenA, tokenB));

        // Set the exchange contract in the mapping
        exchangeContracts[tokenA][tokenB] = exchangeContract;
        exchangeContracts[tokenB][tokenA] = exchangeContract;

        // Set the exchange contract creator in the mapping
        exchangeContractCreators[tokenA][tokenB] = msg.sender;
        exchangeContractCreators[tokenB][tokenA] = msg.sender;

        emit NewExchangeContract(tokenA, tokenB, exchangeContract, msg.sender);
    }

    // Function to get the exchange contract for a token pair
    function getExchangeContract(address tokenA, address tokenB) public view returns (address) {
        return exchangeContracts[tokenA][tokenB];
    }

    // Function to get all exchange contracts
    function getAllExchangeContracts() public view returns (address[] memory) {
        address[] memory exchangeContractsArray = new address[](exchangeContracts.length);
        uint256 index = 0;
        for (address tokenA in exchangeContracts) {
            for (address tokenB in exchangeContracts[tokenA]) {
                exchangeContractsArray[index] = exchangeContracts[tokenA][tokenB];
                index++;
            }
        }
        return exchangeContractsArray;
    }

    // Function to get the creator of an exchange contract
    function getExchangeContractCreator(address tokenA, address tokenB) public view returns (address) {
        return exchangeContractCreators[tokenA][tokenB];
    }
}

contract ExchangeContract {
    address public tokenA;
    address public tokenB;

    constructor(address _tokenA, address _tokenB) public {
        tokenA = _tokenA;
        tokenB = _tokenB;
    }

    // Function to swap tokens
    function swap(address user, uint256 amountA, uint256 amountB) public {
        // Implement the swap logic here
        ERC20(tokenA).transferFrom(user, address(this), amountA);
        ERC20(tokenB).transfer(user, amountB);
    }

    // Function to add liquidity
    function addLiquidity(address user, uint256 amountA, uint256 amountB) public {
        // Implement the add liquidity logic here
        ERC20(tokenA).transferFrom(user, address(this), amountA);
        ERC20(tokenB).transferFrom(user, address(this), amountB);
    }

    // Function to remove liquidity
    function removeLiquidity(address user, uint256 amountA, uint256 amountB) public {
        // Implement the remove liquidity logic here
        ERC20(tokenA).transfer(user, amountA);
        ERC20(tokenB).transfer(user, amountB);
    }
}
