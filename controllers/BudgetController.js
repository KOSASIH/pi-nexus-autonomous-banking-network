// controllers/BudgetController.js
const BudgetService = require("../services/BudgetService");
const Budget = require("../models/Budget");

const createBudget = async (req, res) => {
  const budget = new Budget(req.body);
  await budget.save();
  res.send({ message: "Budget created successfully" });
};

const getBudgetRecommendations = async (req, res) => {
  const budget = await Budget.findById(req.params.id);
  const recommendations =
    await BudgetService.generateBudgetRecommendations(budget);
  res.send(recommendations);
};

module.exports = { createBudget, getBudgetRecommendations };
