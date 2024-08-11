pragma solidity ^0.8.0;

import "./PiCoin.sol";
import "./Treasury.sol";
import "./Oracle.sol";
import "./Math.sol";

contract CollateralizationContract {
    // Mapping of collateral assets to their corresponding balances
    mapping (address => uint256) public collateralBalances;

    // Mapping of collateral assets to their corresponding weights
    mapping (address => uint256) public collateralWeights;

    // Mapping of collateral assets to their corresponding price feeds
    mapping (address => address) public collateralPriceFeeds;

    // Pi Coin contract address
    address public piCoinAddress;

    // Treasury contract address
    address public treasuryAddress;

    // Oracle contract address
    address public oracleAddress;

    // Collateralization ratio (e.g., 150% means 1.5x collateralization)
    uint256 public collateralizationRatio;

    // Minimum collateralization ratio (e.g., 100% means 1x collateralization)
    uint256 public minCollateralizationRatio;

    // Maximum collateralization ratio (e.g., 200% means 2x collateralization)
    uint256 public maxCollateralizationRatio;

    // Collateralization buffer (e.g., 10% means 10% of the collateral value is reserved)
    uint256 public collateralizationBuffer;

    // Event emitted when collateral is deposited
    event CollateralDeposited(address collateralAsset, uint256 amount);

    // Event emitted when collateral is withdrawn
    event CollateralWithdrawn(address collateralAsset, uint256 amount);

    // Event emitted when the collateralization ratio is updated
    event CollateralizationRatioUpdated(uint256 newRatio);

    // Event emitted when the collateralization buffer is updated
    event CollateralizationBufferUpdated(uint256 newBuffer);

    // Event emitted when the minimum collateralization ratio is updated
    event MinCollateralizationRatioUpdated(uint256 newRatio);

    // Event emitted when the maximum collateralization ratio is updated
    event MaxCollateralizationRatioUpdated(uint256 newRatio);

    // Constructor
    constructor(address _piCoinAddress, address _treasuryAddress, address _oracleAddress, uint256 _collateralizationRatio, uint256 _minCollateralizationRatio, uint256 _maxCollateralizationRatio, uint256 _collateralizationBuffer) public {
        piCoinAddress = _piCoinAddress;
        treasuryAddress = _treasuryAddress;
        oracleAddress = _oracleAddress;
        collateralizationRatio = _collateralizationRatio;
        minCollateralizationRatio = _minCollateralizationRatio;
        maxCollateralizationRatio = _maxCollateralizationRatio;
        collateralizationBuffer = _collateralizationBuffer;
    }

    // Function to deposit collateral
    function depositCollateral(address collateralAsset, uint256 amount) public {
        // Check if the collateral asset is supported
        require(collateralWeights[collateralAsset] > 0, "Collateral asset not supported");

        // Update the collateral balance
        collateralBalances[collateralAsset] += amount;

        // Emit the CollateralDeposited event
        emit CollateralDeposited(collateralAsset, amount);

        // Update the collateralization ratio
        updateCollateralizationRatio();
    }

    // Function to withdraw collateral
    function withdrawCollateral(address collateralAsset, uint256 amount) public {
        // Check if the collateral balance is sufficient
        require(collateralBalances[collateralAsset] >= amount, "Insufficient collateral balance");

        // Update the collateral balance
        collateralBalances[collateralAsset] -= amount;

        // Emit the CollateralWithdrawn event
        emit CollateralWithdrawn(collateralAsset, amount);

        // Update the collateralization ratio
        updateCollateralizationRatio();
    }

    // Function to update the collateralization ratio
    function updateCollateralizationRatio() public {
        // Get the total collateral value
        uint256 totalCollateralValue = getTotalCollateralValue();

        // Get the Pi Coin supply
        uint256 piCoinSupply = PiCoin(piCoinAddress).totalSupply();

        // Calculate the new collateralization ratio
        uint256 newRatio = (totalCollateralValue * 100) / piCoinSupply;

        // Check if the new ratio is within the allowed range
        require(newRatio >= minCollateralizationRatio && newRatio <= maxCollateralizationRatio, "Collateralization ratio out of range");

        // Update the collateralization ratio
        collateralizationRatio = newRatio;

        // Emit the CollateralizationRatioUpdated event
        emit CollateralizationRatioUpdated(newRatio);
    }

    // Function to update the collateralization buffer
    function updateCollateralizationBuffer(uint256 newBuffer) public {
        // Check if the new buffer is valid
        require(newBuffer >= 0 && newBuffer <= 100, "Invalid collateralization buffer");

        // Update the collateralization buffer
        collateralizationBuffer = newBuffer;

        // Emit the CollateralizationBufferUpdated event
        emit CollateralizationBufferUpdated(newBuffer);
    }

    // Function to update the minimum collateralization ratio
    function updateMinCollateralizationRatio(uint256 newRatio) public {
        // Check if the new ratio is valid
        require(newRatio >= 0 && newRatio <= 100, "Invalid minimum collateralization ratio");

        // Update the minimum collateralization ratio
        minCollateralizationRatio = newRatio;

        // Emit the MinCollateralizationRatioUpdated event
        emit MinCollateralizationRatioUpdated(newRatio);
    }

    // Function to update the maximum collateralization ratio
    function updateMaxCollateralizationRatio(uint256 newRatio) public {
        // Check if the new ratio is valid
        require(newRatio >= 0 && newRatio <= 100, "Invalid maximum collateralization ratio");

        // Update the maximum collateralization ratio
        maxCollateralizationRatio = newRatio;

        // Emit the MaxCollateralizationRatioUpdated event
        emit MaxCollateralizationRatioUpdated(newRatio);
    }

    // Function to get the total collateral value
    function getTotalCollateralValue() public view returns (uint256) {
        uint256 totalValue = 0;
        for (address collateralAsset in collateralBalances) {
            uint256 price = Oracle(oracleAddress).getPrice(collateralAsset);
            totalValue += collateralBalances[collateralAsset] * price;
        }
        return totalValue;
    }

    // Function to get the Pi Coin supply
    function getPiCoinSupply() public view returns (uint256) {
        return PiCoin(piCoinAddress).totalSupply();
    }

    // Function to check if the collateralization ratio is met
    function isCollateralizationRatioMet() public view returns (bool) {
        uint256 totalCollateralValue = getTotalCollateralValue();
        uint256 piCoinSupply = getPiCoinSupply();
        return totalCollateralValue >= piCoinSupply * collateralizationRatio / 100;
    }

    // Function to check if the collateralization buffer is met
    function isCollateralizationBufferMet() public view returns (bool) {
        uint256 totalCollateralValue = getTotalCollateralValue();
        uint256 piCoinSupply = getPiCoinSupply();
        return totalCollateralValue >= piCoinSupply * (collateralizationRatio + collateralizationBuffer) / 100;
    }

    // Function to get the collateralization ratio
    function getCollateralizationRatio() public view returns (uint256) {
        return collateralizationRatio;
    }

    // Function to get the collateralization buffer
    function getCollateralizationBuffer() public view returns (uint256) {
        return collateralizationBuffer;
    }

    // Function to get the minimum collateralization ratio
    function getMinCollateralizationRatio() public view returns (uint256) {
        return minCollateralizationRatio;
    }

    // Function to get the maximum collateralization ratio
    function getMaxCollateralizationRatio() public view returns (uint256) {
        return maxCollateralizationRatio;
    }
}
