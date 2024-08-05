pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "./WalletContract.sol";

contract DexAppContract {
    using SafeMath for uint256;
    using SafeERC20 for ERC20;

    // Mapping of token addresses to their respective liquidity pools
    mapping (address => mapping (address => uint256)) public liquidityPools;

    // Mapping of user addresses to their respective orders
    mapping (address => mapping (address => uint256)) public orders;

    // Event emitted when a user places an order
    event OrderPlaced(address indexed user, address indexed token, uint256 amount, uint256 price);

    // Event emitted when a user cancels an order
    event OrderCanceled(address indexed user, address indexed token, uint256 amount);

    // Event emitted when a trade is executed
    event TradeExecuted(address indexed buyer, address indexed seller, address indexed token, uint256 amount, uint256 price);

    // Function to place an order
    function placeOrder(address _token, uint256 _amount, uint256 _price) public {
        require(_amount > 0, "Invalid amount");
        require(_price > 0, "Invalid price");
        orders[msg.sender][_token] = _amount;
        emit OrderPlaced(msg.sender, _token, _amount, _price);
    }

    // Function to cancel an order
    function cancelOrder(address _token) public {
        require(orders[msg.sender][_token] > 0, "No order to cancel");
        delete orders[msg.sender][_token];
        emit OrderCanceled(msg.sender, _token, orders[msg.sender][_token]);
    }

    // Function to execute a trade
    function executeTrade(address _buyer, address _seller, address _token, uint256 _amount, uint256 _price) public {
        require(orders[_seller][_token] >= _amount, "Insufficient liquidity");
        require(WalletContract(_seller).balanceOf(_seller, _token) >= _amount, "Insufficient balance");
        WalletContract(_seller).transfer(_buyer, _token, _amount);
        emit TradeExecuted(_buyer, _seller, _token, _amount, _price);
    }

    // Function to add liquidity to a pool
    function addLiquidity(address _token, uint256 _amount) public {
        require(_amount > 0, "Invalid amount");
        liquidityPools[_token][msg.sender] = liquidityPools[_token][msg.sender].add(_amount);
        WalletContract(msg.sender).transfer(address(this), _token, _amount);
    }

    // Function to remove liquidity from a pool
    function removeLiquidity(address _token, uint256 _amount) public {
        require(liquidityPools[_token][msg.sender] >= _amount, "Insufficient liquidity");
        liquidityPools[_token][msg.sender] = liquidityPools[_token][msg.sender].sub(_amount);
        WalletContract(address(this)).transfer(msg.sender, _token, _amount);
    }
}
