pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract Oracle {
    using SafeMath for uint256;

    address public owner;
    mapping (string => uint256) public data;

    event NewData(string indexed key, uint256 value);

    constructor() public {
        owner = msg.sender;
    }

    function updateData(string memory _key, uint256 _value) public onlyOwner {
        data[_key] = _value;
        emit NewData(_key, _value);
    }

    function getData(string memory _key) public view returns (uint256) {
        return data[_key];
    }
}
