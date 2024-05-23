pragma solidity ^0.8.0;

contract PIBank {
    struct Account {
        uint balance;
        address user;
    }

    mapping(address => Account) public accounts;

    event NewTransaction(
        address indexed sender,
        address indexed receiver,
        uint amount
    );

    constructor() {
        addAccount(msg.sender, 0);
    }

    function addAccount(address _user, uint _initialBalance) private {
        accounts[_user] = Account(_initialBalance, _user);
    }

    function deposit() public payable {
        require(msg.value > 0, "No value sent");
        accounts[msg.sender].balance += msg.value;
        emit NewTransaction(msg.sender, address(this), msg.value);
    }

    function withdraw(uint _amount) public {
        require(_amount <= accounts[msg.sender].balance, "Insufficient balance");
        require(msg.sender.call{value: _amount}(""), "Transfer failed");
        accounts[msg.sender].balance -= _amount;
        emit NewTransaction(msg.sender, address(this), _amount);
    }

    function transfer(address _to, uint _amount) public {
        require(_to != address(0), "Invalid address");
        require(_amount <= accounts[msg.sender].balance, "Insufficient balance");

        accounts[_to].balance += _amount;
        accounts[msg.sender].balance -= _amount;

        emit NewTransaction(msg.sender, _to, _amount);
    }

    function getBalance() public view returns (uint) {
        return accounts[msg.sender].balance;
    }
}
