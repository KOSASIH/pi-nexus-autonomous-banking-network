const express = require("express");
const router = express.Router();

const AccountsController = require("./accounts");
const TransfersController = require("./transfers");

router.use("/accounts", AccountsController);
router.use("/transfers", TransfersController);

module.exports = router;
