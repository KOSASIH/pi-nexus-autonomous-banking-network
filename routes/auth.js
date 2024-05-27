import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { User } from "../models/User";

const router = express.Router();

router.post("/login", async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });
  if (!user) return res.status(400).send("Invalid email or password");
  const isPasswordValid = await bcrypt.compare(password, user.password);
  if (!isPasswordValid) {
    return res.status(400).send("Invalid email or password");
  }
  const token = jwt.sign(
    { _id: user._id, email: user.email },
    process.env.SECRET_KEY,
    {
      expiresIn: "1h",
    },
  );
  res.send(token);
});

router.post("/register", async (req, res) => {
  const { email, password, name, role } = req.body;
  const existingUser = await User.findOne({ email });
  if (existingUser) return res.status(400).send("Email already exists");
  const hashedPassword = await bcrypt.hash(password, 10);
  const newUser = new User({ email, password: hashedPassword, name, role });
  await newUser.save();
  res.send("User created successfully");
});

export default router;
