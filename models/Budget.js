// models/Budget.js
const mongoose = require("mongoose");

const budgetSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
  },
  income: {
    type: Number,
    required: true,
  },
  expenses: [
    {
      category: {
        type: String,
        required: true,
      },
      amount: {
        type: Number,
        required: true,
      },
    },
  ],
  forecast: {
    type: Object,
    required: true,
  },
});

module.exports = mongoose.model("Budget", budgetSchema);
