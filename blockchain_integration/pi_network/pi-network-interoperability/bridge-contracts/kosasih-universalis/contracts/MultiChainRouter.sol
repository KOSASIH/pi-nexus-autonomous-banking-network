pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract MultiChainRouter {
    address public kosasihUniversalisBridge;
    address public ethereumBridge;
    address public binanceSmartChainBridge;

    constructor(address _kosasihUniversalisBridge, address _ethereumBridge, address _binanceSmartChainBridge) public {
        kosasihUniversalisBridge = _kosasihUniversalisBridge;
        ethereumBridge = _ethereumBridge;
        binanceSmartChainBridge = _binanceSmartChainBridge;
    }

    function transferTokens(address _token, address _from, address _to, uint256 _amount) public {
        // Determine the destination chain and call the corresponding bridge
        if (_to.chainId == 1) {
            // Ethereum
            EthereumBridge(ethereumBridge).transferTokens(_token, _from, _to, _amount);
        } else if (_to.chainId == 56) {
            // Binance Smart Chain
            BinanceSmartChainBridge(binanceSmartChainBridge).transferTokens(_token, _from, _to, _amount);
        } else {
            // Kosasih Universalis
            KosasihUniversalisBridge(kosasihUniversalisBridge).transferTokens(_token, _from, _to, _amount);
        }
    }
}
