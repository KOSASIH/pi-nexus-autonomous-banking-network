// controllers/PaymentController.js
const PaymentService = require('../services/PaymentService');

const sendMoney = async (req, res) => {
  const fromUser = req.user;
  const toUser = await User.findById(req.body.toUserId);
  const amount = req.body.amount;
  const description = req.body.description;
  try {
    const transaction = await PaymentService.sendMoney(fromUser, toUser, amount, description);
    res.send({ message: 'Money sent successfully' });
  } catch (error) {
    res.status(400).send({ message: error.message });
  }
};

const splitBill = async (req, res) => {
  const users = req.body.users;
  const amount = req.body.amount;
  const description = req.body.description;
  try {
    const transactions = await PaymentService.splitBill(users, amount, description);
    res.send({ message: 'Bill split successfully' });
  } catch (error) {
    res.status(400).send({ message: error.message });
  }
};

module.exports = { sendMoney, splitBill };
