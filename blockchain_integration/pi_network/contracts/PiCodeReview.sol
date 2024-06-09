pragma solidity ^0.8.0;

contract PiCodeReview {
    mapping (address => bool) public reviewedContracts;

    function reviewContract(address _contract) public {
        // Implement code review logic here
        reviewedContracts[_contract] = true;
    }

    function isContractReviewed(address _contract) public view returns (bool) {
        return reviewedContracts[_contract];
    }
}
