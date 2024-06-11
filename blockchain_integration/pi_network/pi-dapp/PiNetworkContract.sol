pragma solidity ^0.8.0;

contract PiNetworkContract {
    mapping (address => uint256) public balances;

    function balanceOf(address _address) public view returns (uint256) {
        return balances[_address];
    }

    function transfer(address _from, address _to, uint256 _amount) public {
        require(balances[_from] >= _amount, "Insufficient balance");
        balances[_from] -= _amount;
        balances[_to] += _amount;
    }

    function sendPiCoins(address _to, uint256 _amount) public {
        transfer(msg.sender, _to, _amount);
    }
}
