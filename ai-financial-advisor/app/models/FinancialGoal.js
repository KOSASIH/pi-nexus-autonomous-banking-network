const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const financialGoalSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: "User" },
  category: { type: String, enum: ["savings", "investment", "retirement"] },
  targetAmount: { type: Number, required: true },
  targetDate: { type: Date, required: true },
  currentAmount: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model("FinancialGoal", financialGoalSchema);
