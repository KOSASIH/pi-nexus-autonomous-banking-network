// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "@openzeppelin/contracts/migrations/Migration.sol";

contract Migrations is Migration {
    uint256 private _latestMigration;

    constructor() {
        _latestMigration = 1;
    }

    function latest() public view returns (uint256) {
        return _latestMigration;
    }

    function upgrade(address newImplementationAddress) public {
        _upgradeToAndCall(newImplementationAddress, _latestMigration);
    }

    function upgradeTo(address newImplementationAddress) public {
        _upgradeTo(newImplementationAddress);
    }

    function _upgradeTo(address newImplementationAddress) internal {
        _writeCheckpoint();
        _latestMigration++;
        _implementation = newImplementationAddress;
    }

    function _upgradeToAndCall(address newImplementationAddress, uint256 data) internal {
        _writeCheckpoint();
        _latestMigration++;
        (bool success, ) = _implementation.delegatecall(abi.encodeWithSelector(_delegatecall.selector, data));
        require(success, "Migration failed");
    }

    function _writeCheckpoint() internal {
        _checkpoints[_latestMigration] = block.chainid;
    }
}
