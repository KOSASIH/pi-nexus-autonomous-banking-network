pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNetworkFactory {
    address private owner;
    string private apiUrl = "https://menepi.com/api";

    constructor() {
        owner = msg.sender;
    }

    function createPiNetwork() public returns (address) {
        PiNetwork piNetwork = new PiNetwork();
        return address(piNetwork);
    }

    function getApiUrl() public view returns (string memory) {
        return apiUrl;
    }
}
