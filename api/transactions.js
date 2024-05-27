// api/transactions.js
const express = require("express");
const router = express.Router();
const mongoose = require("mongoose");
const Transaction = require("../models/Transaction");
const PushNotification = require("../services/PushNotification");

router.post("/track", async (req, res) => {
  const transaction = new Transaction(req.body);
  try {
    await transaction.save();
    // Send push notification for large or suspicious transactions
    if (transaction.amount > 1000 || transaction.isSuspicious) {
      PushNotification.sendNotification(
        transaction.user,
        `Large transaction detected: ${transaction.amount}`,
      );
    }
    res.status(201).send({ message: "Transaction tracked successfully" });
  } catch (error) {
    console.error(error);
    res.status(500).send({ message: "Error tracking transaction" });
  }
});

router.get("/history", async (req, res) => {
  const transactions = await Transaction.find({ user: req.user.id });
  res.send(transactions);
});

module.exports = router;
