pragma solidity ^0.8.0;

import "https://github.com/binance-chain/bsc-bridge-solidity/contracts/BinanceSmartChain.sol";

contract BinanceSmartChainRouter {
    address public bscAddress;
    address public owner;

    constructor() public {
        bscAddress = 0x...; // Binance Smart Chain address
        owner = msg.sender;
    }

    function transferBnb(address _to, uint256 _value) public {
        require(msg.sender == owner, "Only the owner can transfer BNB");
        BinanceSmartChain(bscAddress).transfer(_to, _value);
    }

    function transferToken(address _token, address _to, uint256 _value) public {
        require(msg.sender == owner, "Only the owner can transfer tokens");
        BinanceSmartChain(bscAddress).transferToken(_token, _to, _value);
    }

    function getBnbBalance(address _address) public view returns (uint256) {
        return BinanceSmartChain(bscAddress).getBalance(_address);
    }

    function getTokenBalance(address _token, address _address) public view returns (uint256) {
        return BinanceSmartChain(bscAddress).getTokenBalance(_token, _address);
    }
}
