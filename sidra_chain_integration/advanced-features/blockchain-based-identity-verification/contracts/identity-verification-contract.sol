// identity-verification-contract.sol
pragma solidity ^0.8.10;

contract IdentityVerificationContract {
    address private owner;
    mapping (address => bool) public verifiedUsers;

    constructor() public {
        owner = msg.sender;
    }

    function verifyUser(address userAddress) public {
        require(msg.sender == owner, "Only the owner can verify users");
        verifiedUsers[userAddress] = true;
    }

    function isUserVerified(address userAddress) public view returns (bool) {
        return verifiedUsers[userAddress];
    }
}
