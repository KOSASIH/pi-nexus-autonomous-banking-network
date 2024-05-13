pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MultiChainBank is ERC20, Ownable {
    IMultiChainBankManager public manager;
    IMultiChainOracle public oracle;

    constructor(IMultiChainBankManager manager, IMultiChainOracle oracle) {
        require(manager != address(0), "Invalid manager address");
        require(oracle != address(0), "Invalid oracle address");
        this.manager = manager;
        this.oracle = oracle;
        _mint(msg.sender, 1000000 * (10 ** decimals()));
    }

    function deposit() external payable {
        require(msg.value > 0, "Invalid deposit amount");
        _mint(msg.sender, msg.value * (10 ** decimals()));
    }

    function withdraw(uint256 amount) external {
        require(amount > 0, "Invalid withdrawal amount");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _burn(msg.sender, amount);
        payable(msg.sender).transfer(amount);
    }

    function getBalance() external view override returns (uint256) {
        return balanceOf(msg.sender);
    }

    function transfer(address recipient, uint256 amount) external override {
        require(recipient != address(0), "Invalid recipient address");
        require(amount > 0, "Invalid transfer amount");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _transfer(msg.sender, recipient, amount);
    }

    function approve(address spender, uint256 amount) external override {
        require(spender != address(0), "Invalid spender address");
        require(amount > 0, "Invalid approval amount");
        _approve(msg.sender, spender, amount);
    }

    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowance(owner, spender);
    }

    function totalSupply() external view override returns (uint256) {
        return _totalSupply();
    }

    function addBank(IMultiChainBank bank) external {
        require(msg.sender == manager, "Only the manager can add a bank");
        require(bank != address(0), "Invalid bank address");
        require(bank.totalSupply() > 0, "Bank has no tokens");
        require(bank.balanceOf(address(this)) == bank.totalSupply(), "Bank has insufficient balance");
    }

    function removeBank(IMultiChainBank bank) external {
        require(msg.sender == manager, "Only the manager can remove a bank");
        require(bank != address(0), "Invalid bank address");
        require(bank.totalSupply() > 0, "Bank has no tokens");
        require(bank.balanceOf(address(this)) == bank.totalSupply(), "Bank has insufficient balance");
        _burn(address(bank), bank.totalSupply());
    }

    function getBanks() external view returns (IMultiChainBank[] memory) {
        return manager.getBanks();
    }

    function getPrice(address token) external view returns (uint256) {
        return oracle.getPrice(token);
    }
}
