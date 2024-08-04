pragma solidity ^0.8.0;

contract CognitaToken {
    // Token properties
    string public name;
    string public symbol;
    uint public totalSupply;

    // Mapping of token balances
    mapping (address => uint) public balances;

    // Events
    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);

    // Constructor
    constructor() public {
        name = "Cognita Token";
        symbol = "COG";
        totalSupply = 100000000;
    }

    // Functions
    function transfer(address to, uint value) public {
        // Transfer logic
    }

    function approve(address spender, uint value) public {
        // Approval logic
    }
}
