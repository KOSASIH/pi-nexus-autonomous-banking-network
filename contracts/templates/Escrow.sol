pragma solidity ^0.8.0;

contract Escrow {

    // The escrowed asset
    address public asset;

    // The beneficiary
    address public beneficiary;

    // The amount
    uint public amount;

    // The function to initialize the contract
    constructor(address _asset, address _beneficiary, uint _amount) {
        asset = _asset;
        beneficiary = _beneficiary;
        amount = _amount;
    }

    // The function to release the escrowed asset
    function release() external {
        require(msg.sender == beneficiary, "Only the beneficiary can release the escrowed asset");

        asset.transfer(beneficiary, amount);
    }

}
