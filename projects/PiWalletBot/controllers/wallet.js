import PiCoin from '../models/PiCoin';
import Transaction from '../models/Transaction';
import User from '../models/User';

export async function getBalance(req, res) {
  const userId = req.user._id;
  const user = await User.findById(userId);
  const piCoinBalance = user.piCoinBalance;
  res.json({ balance: piCoinBalance });
}

export async function sendTransaction(req, res) {
  const { to, value } = req.body;
  const userId = req.user._id;
  const user = await User.findById(userId);
  const piCoin = await PiCoin.findOne({ symbol: 'PIC' });
  const transaction = new Transaction({
    piCoinId: piCoin._id,
    from: user._id,
    to,
    value,
  });
  await transaction.save();
  user.piCoinBalance -= value;
  await user.save();
  res.json({ message: 'Transaction sent successfully' });
}

export async function getTransactionHistory(req, res) {
  const userId = req.user._id;
  const transactions = await Transaction.find({ $or: [{ from: userId }, { to: userId }] });
  res.json({ transactions });
}
