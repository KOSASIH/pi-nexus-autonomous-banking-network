const express = require("express");
const router = express.Router();
const { User } = require("../models/user");
const { authenticate } = require("../middleware/authenticate");
const { authorize } = require("../middleware/authorize");
const { validateUser } = require("../validation/user");

// Create a new user
router.post("/", async (req, res) => {
  const { error } = validateUser(req.body);
  if (error) return res.status(400).send(error.details[0].message);

  const user = new User({
    name: req.body.name,
    email: req.body.email,
    password: req.body.password,
  });

  try {
    await user.save();
    res.send(user);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Get all users
router.get("/", authenticate, authorize("admin"), async (req, res) => {
  try {
    const users = await User.find().select("-password");
    res.send(users);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Get a user by ID
router.get("/:id", authenticate, authorize("admin"), async (req, res) => {
  try {
    const user = await User.findById(req.params.id).select("-password");
    if (!user) return res.status(404).send("User not found");
    res.send(user);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Update a user
router.put("/:id", authenticate, authorize("admin"), async (req, res) => {
  const { error } = validateUser(req.body);
  if (error) return res.status(400).send(error.details[0].message);

  try {
    const user = await User.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!user) return res.status(404).send("User not found");
    res.send(user);
    } catch (error) {
    res.status(500).send(error.message);
  }
});

// Delete a user
router.delete("/:id", authenticate, authorize("admin"), async (req, res) => {
  try {
    const user = await User.findByIdAndRemove(req.params.id);
    if (!user) return res.status(404).send("User not found");
    res.send(user);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Login user
router.post("/login", async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(401).send("Invalid email or password");
    const isValid = await user.comparePassword(password);
    if (!isValid) return res.status(401).send("Invalid email or password");
    const token = user.generateAuthToken();
    res.send({ token });
  } catch (error) {
    res.status(500).send(error.message);
  }
});

module.exports = router;
 
