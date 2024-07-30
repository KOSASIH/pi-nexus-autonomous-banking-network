pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract SecurityContract is Ownable {
    using SafeMath for uint256;

    // Mapping of user addresses to their respective access levels
    mapping (address => uint256) public accessLevels;

    // Mapping of user addresses to their respective authentication timestamps
    mapping (address => uint256) public authTimestamps;

    // Event emitted when a user's access level is updated
    event AccessLevelUpdated(address indexed user, uint256 newAccessLevel);

    // Event emitted when a user is authenticated
    event UserAuthenticated(address indexed user, uint256 timestamp);

    // Event emitted when a user's authentication is revoked
    event UserAuthenticationRevoked(address indexed user);

    // Modifier to restrict access to only the owner
    modifier onlyOwner() {
        require(msg.sender == owner(), "Only the owner can call this function");
        _;
    }

    // Function to update a user's access level
    function updateAccessLevel(address _user, uint256 _newAccessLevel) public onlyOwner {
        accessLevels[_user] = _newAccessLevel;
        emit AccessLevelUpdated(_user, _newAccessLevel);
    }

    // Function to authenticate a user
    function authenticateUser(address _user) public {
        authTimestamps[_user] = block.timestamp;
        emit UserAuthenticated(_user, block.timestamp);
    }

    // Function to revoke a user's authentication
    function revokeUserAuthentication(address _user) public onlyOwner {
        delete authTimestamps[_user];
        emit UserAuthenticationRevoked(_user);
    }

    // Function to check if a user is authenticated
    function isAuthenticated(address _user) public view returns (bool) {
        return authTimestamps[_user] != 0;
    }

    // Function to check a user's access level
    function getAccessLevel(address _user) public view returns (uint256) {
        return accessLevels[_user];
    }
}
