pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract PiNexusBankingContractV2 is ERC20, Roles {
    address private _owner;
    uint256 private _totalSupply;
    mapping (address => uint256) public vestingSchedules;

    constructor() public {
        _owner = msg.sender;
        _totalSupply = 100000000; // Initial token supply
    }

    function mint(address _to, uint256 _amount) public onlyMinter {
        _mint(_to, _amount);
    }

    function burn(address _from, uint256 _amount) public onlyBurner {
        _burn(_from, _amount);
    }

    function transfer(address _to, uint256 _amount) public {
        require(hasRole(MINTER_ROLE, msg.sender) || hasRole(BURNER_ROLE, msg.sender), "Only minters or burners can transfer");
        _transfer(msg.sender, _to, _amount);
    }

    function approve(address _spender, uint256 _amount) public {
        require(hasRole(MINTER_ROLE, msg.sender) || hasRole(BURNER_ROLE, msg.sender), "Only minters or burners can approve");
        _approve(msg.sender, _spender, _amount);
    }

    function setVestingSchedule(address _beneficiary, uint256 _amount, uint256 _vestingPeriod) public onlyOwner {
        vestingSchedules[_beneficiary] = _amount;
        // Implement vesting logic
    }

    function lockTokens(address _owner, uint256 _amount, uint256 _lockPeriod) public onlyOwner {
        // Implement token locking logic
    }

    event Transfer(address indexed _from, address indexed _to, uint256 _amount);
    event Approval(address indexed _owner, address indexed _spender, uint256 _amount);
    event VestingScheduleSet(address indexed _beneficiary, uint256 _amount, uint256 _vestingPeriod);
    event TokensLocked(address indexed _owner, uint256 _amount, uint256 _lockPeriod);
}
