// tests/contract-executor.test.js

const { ContractExecutor } = require('../contract-executor');
const { expect } = require('chai');

describe('Contract Executor', () => {
  let contractExecutor;

  beforeEach(() => {
    contractExecutor = new ContractExecutor();
  });

  it('should execute smart contracts', async () => {
    const contractCode = `
      pragma solidity ^0.8.0;

      contract SimpleContract {
        address public owner;
        uint public balance;

        constructor() public {
          owner = msg.sender;
          balance = 0;
        }

        function deposit() public payable {
          balance += msg.value;
        }

        function withdraw(uint amount) public {
          require(msg.sender == owner, 'Only the owner can withdraw');
          balance -= amount;
          msg.sender.transfer(amount);
        }
      }
    `;
    const contract = await contractExecutor.compile(contractCode);
    const instance = await contractExecutor.deploy(contract);
    await instance.deposit({ value: 10 });
    expect(await instance.balance()).to.equal(10);
    await instance.withdraw(5);
    expect(await instance.balance()).to.equal(5);
  });

  it('should throw an error for invalid contract code', async () => {
    const contractCode = `
      pragma solidity ^0.8.0;

      contract InvalidContract {
        invalid syntax
      }
    `;
    try {
      await contractExecutor.compile(contractCode);
      throw new Error('Expected an error to be thrown');
    } catch (error) {
      expect(error).to.be.an.instanceof(Error);
    }
  });
});
