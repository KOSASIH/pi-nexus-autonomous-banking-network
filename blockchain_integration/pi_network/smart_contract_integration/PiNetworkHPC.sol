pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/HPC/HPC.sol";

contract PiNetworkHPC is HPC {
    // Mapping of user addresses to their HPC resources
    mapping (address => HPCResource) public hpcResources;

    // Struct to represent an HPC resource
    struct HPCResource {
        string resourceType;
        uint256 resourceCapacity;
    }

    // Event emitted when a new HPC resource is allocated
    event HPCResourceAllocatedEvent(address indexed user, HPCResource resource);

    // Function to allocate a new HPC resource
    function allocateHPCResource(string memory resourceType, uint256 resourceCapacity) public {
        HPCResource storage resource = hpcResources[msg.sender];
        resource.resourceType = resourceType;
        resource.resourceCapacity = resourceCapacity;
        emit HPCResourceAllocatedEvent(msg.sender, resource);
    }

    // Function to get an HPC resource
    function getHPCResource(address user) public view returns (HPCResource memory) {
        return hpcResources[user];
    }
}
