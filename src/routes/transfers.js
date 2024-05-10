const express = require('express')
const router = express.Router()

const TransferController = require('../controllers/transfers')
const authenticate = require('../middleware/auth')
const createTransaction = require('../middleware/network')

router.post('/', authenticate, async (req, res) => {
  try {
    const transfer = await TransferController.create(req, res)
    const transactionHash = await createTransaction(
      req.user.wallet_address,
      req.body.receiver_account_id,
      req.body.amount
    )
    transfer.transaction_hash = transactionHash
    await transfer.save()
    res.json({ message: 'Transfer successful', transfer })
  } catch (error) {
    res.status(400).json({ message: error.message })
  }
})

module.exports = router
