// auth/mfa.js
const express = require("express");
const router = express.Router();
const mongoose = require("mongoose");
const User = require("../models/User");
const otpGenerator = require("otp-generator");
const speakeasy = require("speakeasy");
const smartCardAuth = require("../services/SmartCardAuth");

router.post("/mfa/setup", async (req, res) => {
  const user = req.user;
  const mfaMethod = req.body.mfaMethod;

  switch (mfaMethod) {
    case "otp":
      const otp = otpGenerator.generate(6);
      user.mfa.otp.secret = otp;
      await user.save();
      res.send({ message: "OTP setup successfully" });
      break;
    case "authenticator":
      const authenticatorSecret = speakeasy.generateSecret();
      user.mfa.authenticator.secret = authenticatorSecret.base32;
      await user.save();
      res.send({ message: "Authenticator setup successfully" });
      break;
    case "smartCard":
      const smartCardToken = await smartCardAuth.generateToken(user);
      user.mfa.smartCard.token = smartCardToken;
      await user.save();
      res.send({ message: "Smart card setup successfully" });
      break;
    default:
      res.status(400).send({ message: "Invalid MFA method" });
  }
});

router.post("/mfa/verify", async (req, res) => {
  const user = req.user;
  const mfaMethod = req.body.mfaMethod;
  const verificationCode = req.body.verificationCode;

  switch (mfaMethod) {
    case "otp":
      if (user.mfa.otp.secret === verificationCode) {
        res.send({ message: "OTP verified successfully" });
      } else {
        res.status(401).send({ message: "Invalid OTP" });
      }
      break;
    case "authenticator":
      const authenticatorToken = speakeasy.totp.verify(
        verificationCode,
        user.mfa.authenticator.secret,
      );
      if (authenticatorToken) {
        res.send({ message: "Authenticator verified successfully" });
      } else {
        res.status(401).send({ message: "Invalid authenticator token" });
      }
      break;
    case "smartCard":
      const smartCardVerified = await smartCardAuth.verifyToken(
        user,
        verificationCode,
      );
      if (smartCardVerified) {
        res.send({ message: "Smart card verified successfully" });
      } else {
        res.status(401).send({ message: "Invalid smart card token" });
      }
      break;
    default:
      res.status(400).send({ message: "Invalid MFA method" });
  }
});

module.exports = router;
