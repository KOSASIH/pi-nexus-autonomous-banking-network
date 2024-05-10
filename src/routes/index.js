const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.json({ message: "Welcome to the PI Nexus Autonomous Banking Network!" });
});

module.exports = router;
