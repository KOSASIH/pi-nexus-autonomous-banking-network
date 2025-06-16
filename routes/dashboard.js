import express from "express";
import { Account } from "../models/Account";
import { Transaction } from "../models/Transaction";

const router = express.Router();

router.get("/accounts", async (req, res) => {
  const accounts = await Account.find({ userId: req.user._id });
  res.send(accounts);
});

router.post("/transactions", async (req, res) => {
  const { accountId, amount, type } = req.body;
  const account = await Account.findOne({
    _id: accountId,
    userId: req.user._id,
  });
  if (!account) return res.status(400).send("Invalid account");
  if (type === "withdrawal" && account.balance < amount) {
    return res.status(400).send("Insufficient balance");
  }
  const newTransaction = new Transaction({ accountId, amount, type });
  await newTransaction.save();
  if (type === "withdrawal") {
    account.balance -= amount;
    await account.save();
  }
  res.send(newTransaction);
});

export default router;
