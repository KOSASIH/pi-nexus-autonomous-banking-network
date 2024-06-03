pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract PiToken is ERC20, Ownable {
    using SafeMath for uint256;
    using Address for address;

    // Token metadata
    string public constant name = "Pi Token";
    string public constant symbol = "PI";
    uint8 public constant decimals = 18;

    // Token supply
    uint256 public totalSupply;
    uint256 public maxSupply;

    // Token minting and burning
    uint256 public mintingCap;
    uint256 public burningCap;
    mapping(address => uint256) public mintingAllowances;
    mapping(address => uint256) public burningAllowances;

    // Transfer restrictions
    mapping(address => bool) public transferWhitelist;
    mapping(address => bool) public transferBlacklist;

    // Vesting and locking
    struct VestingSchedule {
        uint256 amount;
        uint256 startTime;
        uint256 endTime;
        uint256 cliff;
        uint256 duration;
    }
    mapping(address => VestingSchedule[]) public vestingSchedules;
    mapping(address => uint256) public lockedBalances;

    // Events
    event Mint(address indexed recipient, uint256 amount);
    event Burn(address indexed owner, uint256 amount);
    event TransferRestricted(address indexed sender, address indexed recipient, uint256 amount);
    event VestingScheduleCreated(address indexed beneficiary, uint256 amount, uint256 startTime, uint256 endTime);
    event VestingScheduleUpdated(address indexed beneficiary, uint256 amount, uint256 startTime, uint256 endTime);
    event LockedBalanceUpdated(address indexed owner, uint256 amount);

    /**
     * @dev Initializes the Pi Token contract with the specified max supply and minting/burning caps.
     * @param _maxSupply The maximum total supply of Pi Tokens.
     * @param _mintingCap The maximum amount of Pi Tokens that can be minted.
     * @param _burningCap The maximum amount of Pi Tokens that can be burned.
     */
    constructor(uint256 _maxSupply, uint256 _mintingCap, uint256 _burningCap) public {
        maxSupply = _maxSupply;
        mintingCap = _mintingCap;
        burningCap = _burningCap;
        totalSupply = 0;
    }

    /**
     * @dev Mints a specified amount of Pi Tokens to the specified recipient.
     * @param _recipient The address that will receive the minted Pi Tokens.
     * @param _amount The amount of Pi Tokens to mint.
     */
    function mint(address _recipient, uint256 _amount) public onlyOwner {
        require(_amount <= mintingCap, "Minting cap exceeded");
        totalSupply = totalSupply.add(_amount);
        balances[_recipient] = balances[_recipient].add(_amount);
        emit Mint(_recipient, _amount);
    }

    /**
     * @dev Burns a specified amount of Pi Tokens from the specified owner.
     * @param _owner The address that owns the Pi Tokens to burn.
     * @param _amount The amount of Pi Tokens to burn.
     */
    function burn(address _owner, uint256 _amount) public onlyOwner {
        require(_amount <= burningCap, "Burning cap exceeded");
        totalSupply = totalSupply.sub(_amount);
        balances[_owner] = balances[_owner].sub(_amount);
        emit Burn(_owner, _amount);
    }

    /**
     * @dev Transfers a specified amount of Pi Tokens from the specified sender to the specified recipient.
     * @param _sender The address that owns the Pi Tokens to transfer.
     * @param _recipient The address that will receive the transferred Pi Tokens.
     * @param _amount The amount of Pi Tokens to transfer.
     */
    function transfer(address _sender, address _recipient, uint256 _amount) public {
        require(transferWhitelist[_sender] ||!transferBlacklist[_sender], "Transfer restricted");
        require(balances[_sender] >= _amount, "Insufficient balance");
        balances[_sender] = balances[_sender].sub(_amount);
        balances[_recipient] = balances[_recipient].add(_amount);
        emit Transfer(_sender, _recipient, _amount);
    }

    /**
     * @dev Creates a new vesting schedule for the specified beneficiary.
     * @param _beneficiary The address that will receive the vested Pi Tokens.
     * @param _amount Theamount of Pi Tokens to vest.
     * @param _startTime The start time of the vesting schedule.
     * @param _endTime The end time of the vesting schedule.
     * @param _cliff The cliff time of the vesting schedule.
     * @param _duration The duration of the vesting schedule.
     */
    function createVestingSchedule(address _beneficiary, uint256 _amount, uint256 _startTime, uint256 _endTime, uint256 _cliff, uint256 _duration) public onlyOwner {
        require(_startTime > block.timestamp, "Invalid start time");
        require(_endTime > _startTime, "Invalid end time");
        require(_cliff > 0, "Invalid cliff time");
        require(_duration > 0, "Invalid duration");
        require(_amount > 0, "Invalid amount");
        require(totalSupply + _amount <= maxSupply, "Max supply exceeded");

        VestingSchedule memory schedule;
        schedule.amount = _amount;
        schedule.startTime = _startTime;
        schedule.endTime = _endTime;
        schedule.cliff = _cliff;
        schedule.duration = _duration;

        vestingSchedules[_beneficiary].push(schedule);
        lockedBalances[_beneficiary] = lockedBalances[_beneficiary].add(_amount);

        emit VestingScheduleCreated(_beneficiary, _amount, _startTime, _endTime);
    }

    /**
     * @dev Updates the vesting schedule for the specified beneficiary.
     * @param _beneficiary The address that will receive the vested Pi Tokens.
     * @param _amount The updated amount of Pi Tokens to vest.
     * @param _startTime The updated start time of the vesting schedule.
     * @param _endTime The updated end time of the vesting schedule.
     */
    function updateVestingSchedule(address _beneficiary, uint256 _amount, uint256 _startTime, uint256 _endTime) public onlyOwner {
        require(vestingSchedules[_beneficiary].length > 0, "No vesting schedule found");
        require(_startTime > block.timestamp, "Invalid start time");
        require(_endTime > _startTime, "Invalid end time");

        VestingSchedule storage schedule = vestingSchedules[_beneficiary][0];
        schedule.amount = _amount;
        schedule.startTime = _startTime;
        schedule.endTime = _endTime;

        emit VestingScheduleUpdated(_beneficiary, _amount, _startTime, _endTime);
    }

    /**
     * @dev Returns the vested amount of Pi Tokens for the specified beneficiary.
     * @param _beneficiary The address that will receive the vested Pi Tokens.
     * @return The vested amount of Pi Tokens.
     */
    function getVestedAmount(address _beneficiary) public view returns (uint256) {
        require(vestingSchedules[_beneficiary].length > 0, "No vesting schedule found");

        VestingSchedule storage schedule = vestingSchedules[_beneficiary][0];
        uint256 elapsedTime = block.timestamp - schedule.startTime;
        uint256 cliffTime = schedule.cliff;
        uint256 duration = schedule.duration;

        if (elapsedTime < cliffTime) {
            return 0;
        }

        uint256 remainingTime = duration - (elapsedTime - cliffTime);
        uint256 vestedAmount = (schedule.amount * remainingTime) / duration;

        return vestedAmount;
    }

    /**
     * @dev Returns the unlocked amount of Pi Tokens for the specified beneficiary.
     * @param _beneficiary The address that will receive the unlocked Pi Tokens.
     * @return The unlocked amount of Pi Tokens.
     */
    function getUnlockedAmount(address _beneficiary) public view returns (uint256) {
        require(vestingSchedules[_beneficiary].length > 0, "No vesting schedulefound");

        VestingSchedule storage schedule = vestingSchedules[_beneficiary][0];
        uint256 elapsedTime = block.timestamp - schedule.startTime;
        uint256 cliffTime = schedule.cliff;
        uint256 duration = schedule.duration;

        if (elapsedTime < cliffTime) {
            return 0;
        }

        uint256 remainingTime = duration - (elapsedTime - cliffTime);
        uint256 unlockedAmount = schedule.amount - (schedule.amount * remainingTime) / duration;

        return unlockedAmount;
    }
}
