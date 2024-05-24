// services/PaymentService.js
const Wallet = require("../models/Wallet");
const Transaction = require("../models/Transaction");

const sendMoney = async (fromUser, toUser, amount, description) => {
  const fromWallet = await Wallet.findOne({ user: fromUser });
  const toWallet = await Wallet.findOne({ user: toUser });
  if (fromWallet.balance >= amount) {
    const transaction = new Transaction({
      from: fromWallet,
      to: toWallet,
      amount,
      description,
    });
    await transaction.save();
    fromWallet.balance -= amount;
    toWallet.balance += amount;
    await fromWallet.save();
    await toWallet.save();
    return transaction;
  } else {
    throw new Error("Insufficient balance");
  }
};

const splitBill = async (users, amount, description) => {
  const transactions = [];
  for (const user of users) {
    const wallet = await Wallet.findOne({ user });
    const transaction = await sendMoney(
      wallet.user,
      users[0],
      amount / users.length,
      description,
    );
    transactions.push(transaction);
  }
  return transactions;
};

module.exports = { sendMoney, splitBill };
