const Account = require('../models/account')

exports.list = async (req, res) => {
  const accounts = await Account.find({ owner: req.user._id })
  res.json(accounts)
}

exports.create = async (req, res) => {
  const account = new Account({
    account_name: req.body.account_name,
    initial_balance: req.body.initial_balance,
    owner: req.user._id
  })
  await account.save()
  res.json({ message: 'Account created!', account })
}
