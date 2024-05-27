// routes/gamification.js
const express = require("express");
const router = express.Router();
const GamificationController = require("../controllers/GamificationController");

router.post("/save-money", GamificationController.saveMoney);
router.post("/budget-successfully", GamificationController.budgetSuccessfully);
router.post(
  "/make-low-carbon-transaction",
  GamificationController.makeLowCarbonTransaction,
);
router.post("/redeem-reward", GamificationController.redeemReward);

module.exports = router;
