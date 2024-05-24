// models/User.js
const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  // ...
  mfa: {
    otp: {
      secret: String,
    },
    authenticator: {
      secret: String,
    },
    smartCard: {
      token: String,
    },
  },
});

module.exports = mongoose.model("User", userSchema);
