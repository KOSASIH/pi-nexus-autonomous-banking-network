pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNexusRouter is SafeERC20 {
    // Router properties
    address public piNexusToken;
    address public piNexusStaking;
    address public piNexusLending;
    address public piNexusGovernance;

    // Router constructor
    constructor() public {
        piNexusToken = address(new PiNexusToken());
        piNexusStaking = address(new PiNexusStaking());
        piNexusLending = address(new PiNexusLending());
        piNexusGovernance = address(new PiNexusGovernance());
    }

    // Router functions
    function routeTokens(address from, address to, uint256 amount) public {
        // Route tokens between addresses
        _routeTokens(from, to, amount);
    }

    function routeStaking(address from, uint256 amount) public {
        // Route staking tokens to staking contract
        _routeStaking(from, amount);
    }

    function routeLending(address from, uint256 amount) public {
        // Route lending tokens to lending contract
        _routeLending(from, amount);
    }

    function routeGovernance(address from, uint256 amount) public {
        // Route governance tokens to governance contract
        _routeGovernance(from, amount);
    }
}
