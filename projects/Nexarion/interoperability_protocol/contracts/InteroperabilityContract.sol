pragma solidity ^0.8.0;

import "./ERC20.sol";

contract InteroperabilityContract {
    address public owner;
    mapping (address => ERC20) public tokens;

    event TokenAdded(address indexed token);
    event TokenRemoved(address indexed token);

    constructor() public {
        owner = msg.sender;
    }

    function addToken(address _token) public {
        require(msg.sender == owner, "Only the owner can add tokens");
        tokens[_token] = ERC20(_token);
        emit TokenAdded(_token);
    }

    function removeToken(address _token) public {
        require(msg.sender == owner, "Only the owner can remove tokens");
        delete tokens[_token];
        emit TokenRemoved(_token);
    }

    function transferToken(address _token, address _to, uint _value) public {
        require(tokens[_token] != address(0), "Token not found");
        tokens[_token].transfer(_to, _value);
    }

    function approveToken(address _token, address _spender, uint _value) public {
        require(tokens[_token] != address(0), "Token not found");
        tokens[_token].approve(_spender, _value);
    }

    function transferTokenFrom(address _token, address _from, address _to, uint _value) public {
        require(tokens[_token] != address(0), "Token not found");
        tokens[_token].transferFrom(_from, _to, _value);
    }
}
