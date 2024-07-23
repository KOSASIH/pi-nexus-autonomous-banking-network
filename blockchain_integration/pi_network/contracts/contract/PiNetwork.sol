pragma solidity ^0.8.0;

contract PiNetwork {
    address private owner;
    string private apiUrl = "https://menepi.com/api";

    constructor() {
        owner = msg.sender;
    }

    function getApiUrl() public view returns (string memory) {
        return apiUrl;
    }
}
