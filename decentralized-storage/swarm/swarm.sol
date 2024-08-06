// swarm.sol
pragma solidity ^0.8.0;

import "https://github.com/ethersphere/swarm-solidity/blob/master/contracts/Swarm.sol";

contract SwarmStorage {
    Swarm public swarm;

    constructor(address _swarmAddress) public {
        swarm = Swarm(_swarmAddress);
    }

    function uploadFile(bytes memory file) public {
        swarm.upload(file);
    }

    function getFile(bytes32 hash) public view returns (bytes memory) {
        return swarm.getFile(hash);
    }

    function uploadDirectory(bytes memory directory) public {
        swarm.uploadDirectory(directory);
    }

    function getDirectory(bytes32 hash) public view returns (bytes memory) {
        return swarm.getDirectory(hash);
    }
}
