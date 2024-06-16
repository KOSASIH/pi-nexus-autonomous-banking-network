pragma solidity ^0.8.0;

library PIBankUtils {
    function calculateInterestRate(uint256 _balance, uint256 _interestRate) internal pure returns (uint256) {
        return _balance * _interestRate / 100;
    }

    function generateRandomNumber(uint256 _seed) internal view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(_seed, block.timestamp)));
    }

    function encodeData(string memory _data) internal pure returns (bytes memory) {
        return abi.encodePacked(_data);
    }
}
