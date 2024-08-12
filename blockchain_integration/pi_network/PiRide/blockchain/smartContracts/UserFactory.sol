pragma solidity ^0.8.0;

import "./UserContract.sol";

contract UserFactory {
    address[] public userContracts;

    event NewUserContract(address userContract);

    function createUserContract() public {
        UserContract userContract = new UserContract();
        userContracts.push(address(userContract));
        emit NewUserContract(address(userContract));
    }

    function getUserContracts() public view returns (address[] memory) {
        return userContracts;
    }
}
