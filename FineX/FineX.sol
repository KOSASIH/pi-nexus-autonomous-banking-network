pragma solidity ^0.8.0;

contract FineX {
    address private owner;
    uint256 public totalFines;
    uint256 public totalPaid;
    mapping (address => uint256) public fines;

    constructor() {
        owner = msg.sender;
        totalFines = 0;
        totalPaid = 0;
    }

    function issueFine(address _offender, uint256 _amount) public {
        require(msg.sender == owner, "Only the owner can issue fines");
        fines[_offender] += _amount;
        totalFines += _amount;
    }

    function payFine() public payable {
        require(fines[msg.sender] > 0, "You don't have any fines to pay");
        uint256 amountToPay = fines[msg.sender];
        fines[msg.sender] = 0;
        totalPaid += amountToPay;
        payable(msg.sender).transfer(amountToPay);
    }

    function getFineBalance(address _offender) public view returns (uint256) {
        return fines[_offender];
    }

    function getTotalFines() public view returns (uint256) {
        return totalFines;
    }

    function getTotalPaid() public view returns (uint256) {
        return totalPaid;
    }
}
