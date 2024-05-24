// routes/payment.js
const express = require("express");
const router = express.Router();
const PaymentController = require("../controllers/PaymentController");

router.post("/send-money", PaymentController.sendMoney);
router.post("/split-bill", PaymentController.splitBill);

module.exports = router;
