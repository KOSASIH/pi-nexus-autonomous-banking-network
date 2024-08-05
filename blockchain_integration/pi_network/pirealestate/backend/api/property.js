const express = require("express");
const router = express.Router();
const { Property } = require("../models/property");
const { User } = require("../models/user");
const { authenticate } = require("../middleware/authenticate");
const { authorize } = require("../middleware/authorize");
const { validateProperty } = require("../validation/property");

// Create a new property
router.post("/", authenticate, authorize("admin"), async (req, res) => {
  const { error } = validateProperty(req.body);
  if (error) return res.status(400).send(error.details[0].message);

  const property = new Property({
    name: req.body.name,
    description: req.body.description,
    location: req.body.location,
    price: req.body.price,
    owner: req.user._id,
  });

  try {
    await property.save();
    res.send(property);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Get all properties
router.get("/", async (req, res) => {
  try {
    const properties = await Property.find().populate("owner", "_id name email");
    res.send(properties);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Get a property by ID
router.get("/:id", async (req, res) => {
  try {
    const property = await Property.findById(req.params.id).populate("owner", "_id name email");
    if (!property) return res.status(404).send("Property not found");
    res.send(property);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Update a property
router.put("/:id", authenticate, authorize("admin"), async (req, res) => {
  const { error } = validateProperty(req.body);
  if (error) return res.status(400).send(error.details[0].message);

  try {
    const property = await Property.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!property) return res.status(404).send("Property not found");
    res.send(property);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Delete a property
router.delete("/:id", authenticate, authorize("admin"), async (req, res) => {
  try {
    const property = await Property.findByIdAndRemove(req.params.id);
    if (!property) return res.status(404).send("Property not found");
    res.send(property);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

module.exports = router;
