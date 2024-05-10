const Transfer = require('../models/transfer')
const Account = require('../models/account')

exports.create = async (req, res) => {
  const senderAccount = await Account.findById(req.body.sender_account_id)
  const receiverAccount = await Account.findById(req.body.receiver_account_id)

  if (!senderAccount || !receiverAccount) {
    return res.status(400).json({ message: 'Account not found' })
  }

  if (senderAccount.balance < req.body.amount) {
    return res.status(400).json({ message: 'Insufficient balance' })
  }

  const transfer = new Transfer({
    sender_account_id: req.body.sender_account_id,
    receiver_account_id: req.body.receiver_account_id,
    amount: req.body.amount
  })

  await transfer.save()

  senderAccount.balance -= req.body.amount
  receiverAccount.balance += req.body.amount

  await senderAccount.save()
  await receiverAccount.save()

  res.json({ message: 'Transfer successful', transfer })
}
