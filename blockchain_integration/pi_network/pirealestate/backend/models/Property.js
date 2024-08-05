const mongoose = require("mongoose");
const { Schema } = mongoose;
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

const propertySchema = new Schema({
  name: {
    type: String,
    required: true,
    trim: true,
  },
  description: {
    type: String,
    required: true,
    trim: true,
  },
  location: {
    type: String,
    required: true,
    trim: true,
  },
  price: {
    type: Number,
    required: true,
  },
  owner: {
    type: Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

propertySchema.methods.generateToken = function () {
  const property = this;
  const token = jwt.sign({ _id: property._id }, process.env.JWT_SECRET, {
    expiresIn: "1h",
  });
  return token;
};

propertySchema.statics.findByToken = function (token) {
  const Property = this;
  let decoded;
  try {
    decoded = jwt.verify(token, process.env.JWT_SECRET);
  } catch (error) {
    return Promise.reject(error);
  }
  return Property.findOne({ _id: decoded._id });
};

const Property = mongoose.model("Property", propertySchema);

module.exports = Property;
