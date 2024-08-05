pragma solidity ^0.8.0;

contract CognitaToken {
    // Token properties
    string public name;
    string public symbol;
    uint public totalSupply;

    // Mapping of token balances
    mapping (address => uint) public balances;

    // Mapping of token allowances
    mapping (address => mapping (address => uint)) public allowances;

    // Event emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint value);

    // Event emitted when tokens are burned
    event Burn(address indexed from, uint value);

    // Event emitted when dividends are distributed
    event Dividend(address indexed from, uint value);

    // Constructor function
    constructor() public {
        name = "Cognita Token";
        symbol = "COG";
        totalSupply = 100000000;
        balances[msg.sender] = totalSupply;
    }

    // Function to transfer tokens
    function transfer(address to, uint value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    // Function to burn tokens
    function burn(uint value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        totalSupply -= value;
        emit Burn(msg.sender, value);
    }

    // Function to distribute dividends
    function distributeDividend(uint value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        // Distribute dividend to all token holders
        for (address holder in balances) {
            balances[holder] += value / totalSupply;
        }
        emit Dividend(msg.sender, value);
    }
}
