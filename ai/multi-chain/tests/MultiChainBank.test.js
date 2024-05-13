const { expect } = require('chai')
const { ethers } = require('hardhat')

describe('MultiChainBank', function () {
  let multiChainBank
  let owner
  let user1
  let user2

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners()
    const MultiChainBank = await ethers.getContractFactory('MultiChainBank')
    multiChainBank = await MultiChainBank.deploy()
  })

  describe('deployment', function () {
    it('should set the right owner', async function () {
      expect(await multiChainBank.owner()).to.equal(owner.address)
    })

    it('should assign the total supply of tokens to the owner', async function () {
      const ownerBalance = await multiChainBank.balanceOf(owner.address)
      expect(await multiChainBank.totalSupply()).to.equal(ownerBalance)
    })
  })

  describe('transactions', function () {
    it('should transfer tokens between accounts', async function () {
      await multiChainBank.transfer(user1.address, 50)
      const user1Balance = await multiChainBank.balanceOf(user1.address)
      expect(user1Balance).to.equal(50)

      await user1.sendTransaction({ to: multiChainBank.address, value: 50 })
      const user1TokenBalance = await multiChainBank.balanceOf(user1.address)
      expect(user1TokenBalance).to.equal(100)

      await multiChainBank.transfer(user2.address, 50)
      const user2TokenBalance = await multiChainBank.balanceOf(user2.address)
      expect(user2TokenBalance).to.equal(50)
    })

    it('should fail if sender doesn’t have enough tokens', async function () {
      const initialOwnerBalance = await multiChainBank.balanceOf(owner.address)

      await expect(
        multiChainBank.transfer(user1.address, 1)
      ).to.be.revertedWith('MultiChainBank: transfer amount exceeds balance')

      expect(await multiChainBank.balanceOf(owner.address)).to.equal(
        initialOwnerBalance
      )
    })

    it('should update balances after transfers', async function () {
      const initialOwnerBalance = await multiChainBank.balanceOf(owner.address)

      await multiChainBank.transfer(user1.address, 50)

      const finalOwnerBalance = await multiChainBank.balanceOf(owner.address)
      expect(finalOwnerBalance).to.equal(initialOwnerBalance - 50)

      const user1Balance = await multiChainBank.balanceOf(user1.address)
      expect(user1Balance).to.equal(50)
    })
  })
})
