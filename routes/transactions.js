// routes/transactions.js
const express = require("express");
const router = express.Router();
const transactionController = require("../controllers/TransactionController");

router.post("/track", transactionController.trackTransaction);
router.get("/history", transactionController.getTransactionHistory);

module.exports = router;
