const { expect } = require('chai')
const { ethers } = require('hardhat')

describe('MultiChainExchange', function () {
  let multiChainExchange
  let owner
  let user1
  let user2
  let multiChainToken
  let multiChainOracle

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners()
    const MultiChainToken = await ethers.getContractFactory('MultiChainToken')
    multiChainToken = await MultiChainToken.deploy()
    const MultiChainOracle =
      await ethers.getContractFactory('MultiChainOracle')
    multiChainOracle = await MultiChainOracle.deploy()
    const MultiChainExchange =
      await ethers.getContractFactory('MultiChainExchange')
    multiChainExchange = await MultiChainExchange.deploy(
      multiChainToken.address,
      multiChainOracle.address
    )
  })

  describe('deployment', function () {
    it('should set the right token and oracle addresses', async function () {
      expect(await multiChainExchange.token()).to.equal(
        multiChainToken.address
      )
      expect(await multiChainExchange.oracle()).to.equal(
        multiChainOracle.address
      )
    })
  })

  describe('exchange rates', function () {
    it('should allow the owner to set exchange rates', async function () {
      await multiChainOracle.setPrice('token1', 100)
      await multiChainExchange.setExchangeRate('token1', 2)
      const rate = await multiChainExchange.getExchangeRate('token1')
      expect(rate).to.equal(2)
    })

    it('should fail if the sender is not the owner', async function () {
      await expect(
        multiChainExchange.connect(user1).setExchangeRate('token1', 2)
      ).to.be.revertedWith('Ownable: caller is not the owner')
    })
  })

  describe('exchanges', function () {
    it('should allow users to exchange tokens', async function () {
      await multiChainToken.transfer(user1.address, 1000)
      await multiChainOracle.setPrice('token1', 100)
      await multiChainExchange.setExchangeRate('token1', 2)
      await multiChainExchange.connect(user1).exchange(100, 'token1')
      const user1Balance = await multiChainToken.balanceOf(user1.address)
      expect(user1Balance).to.equal(900)
    })

    it('should fail if the user has insufficient balance', async function () {
      await multiChainOracle.setPrice('token1', 100)
      await multiChainExchange.setExchangeRate('token1', 2)
      await expect(
        multiChainExchange.connect(user1).exchange(100, 'token1')
      ).to.be.revertedWith('MultiChainExchange: insufficient balance')
    })

    it('should fail if the oracle price is not set', async function () {
      await multiChainToken.transfer(user1.address, 1000)
      await expect(
        multiChainExchange.connect(user1).exchange(100, 'token1')
      ).to.be.revertedWith('MultiChainExchange: oracle price not set')
    })

    it('should fail if the exchange rate is not set', async function () {
      await multiChainToken.transfer(user1.address, 1000)
      await multiChainOracle.setPrice('token1', 100)
      await expect(
        multiChainExchange.connect(user1).exchange(100, 'token1')
      ).to.be.revertedWith('MultiChainExchange: exchange rate not set')
    })
  })
})
