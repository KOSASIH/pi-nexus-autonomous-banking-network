pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract BinanceSmartChainBridge {
    address public kosasihUniversalisBridge;

    constructor(address _kosasihUniversalisBridge) public {
        kosasihUniversalisBridge = _kosasihUniversalisBridge;
    }

    function transferTokens(address _token, address _from, address _to, uint256 _amount) public {
        // Call the Kosasih Universalis bridge to transfer tokens
        KosasihUniversalisBridge(kosasihUniversalisBridge).transferTokens(_token, _from, _to, _amount);
    }
}
