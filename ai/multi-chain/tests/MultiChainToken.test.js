const { expect } = require('chai')
const { ethers } = require('hardhat')

describe('MultiChainToken', function () {
  let multiChainToken
  let owner
  let user1
  let user2

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners()
    const MultiChainToken = await ethers.getContractFactory('MultiChainToken')
    multiChainToken = await MultiChainToken.deploy()
  })

  describe('deployment', function () {
    it('should set the right owner', async function () {
      expect(await multiChainToken.owner()).to.equal(owner.address)
    })

    it('should assign the total supply of tokens to the owner', async function () {
      const ownerBalance = await multiChainToken.balanceOf(owner.address)
      expect(await multiChainToken.totalSupply()).to.equal(ownerBalance)
    })
  })

  describe('transactions', function () {
    it('should transfer tokens between accounts', async function () {
      await multiChainToken.transfer(user1.address, 50)
      const user1Balance = await multiChainToken.balanceOf(user1.address)
      expect(user1Balance).to.equal(50)

      await user1.sendTransaction({ to: multiChainToken.address, value: 50 })
      const user1TokenBalance = await multiChainToken.balanceOf(user1.address)
      expect(user1TokenBalance).to.equal(100)

      await multiChainToken.transfer(user2.address, 50)
      const user2TokenBalance = await multiChainToken.balanceOf(user2.address)
      expect(user2TokenBalance).to.equal(50)
    })

    it('should fail if sender doesn’t have enough tokens', async function () {
      const initialOwnerBalance = await multiChainToken.balanceOf(
        owner.address
      )

      await expect(
        multiChainToken.transfer(user1.address, 1)
      ).to.be.revertedWith('MultiChainToken: transfer amount exceeds balance')

      expect(await multiChainToken.balanceOf(owner.address)).to.equal(
        initialOwnerBalance
      )
    })

    it('should update balances after transfers', async function () {
      const initialOwnerBalance = await multiChainToken.balanceOf(
        owner.address
      )

      await multiChainToken.transfer(user1.address, 50)

      const finalOwnerBalance = await multiChainToken.balanceOf(owner.address)
      expect(finalOwnerBalance).to.equal(initialOwnerBalance - 50)

      const user1Balance = await multiChainToken.balanceOf(user1.address)
      expect(user1Balance).to.equal(50)
    })
  })
})
