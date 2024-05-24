// routes/budget.js
const express = require('express');
const router = express.Router();
const BudgetController = require('../controllers/BudgetController');

router.post('/create', BudgetController.createBudget);
router.get('/:id/recommendations', BudgetController.getBudgetRecommendations);

module.exports = router;
