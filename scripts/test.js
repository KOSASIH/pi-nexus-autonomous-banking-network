const { expect } = require('chai')

const { ethers } = require('hardhat')

describe('MyContract', function () {
  let myContract

  beforeEach(async function () {
    const MyContract = await ethers.getContractFactory('MyContract')
    myContract = await MyContract.deploy()
    await myContract.deployed()
  })

  describe('constructor', function () {
    it('should set the initial value', async function () {
      const initialValue = await myContract.getValue()
      expect(initialValue).to.equal(42)
    })
  })

  describe('setValue', function () {
    it('should set the value', async function () {
      await myContract.setValue(43)
      const value = await myContract.getValue()
      expect(value).to.equal(43)
    })
  })
})
