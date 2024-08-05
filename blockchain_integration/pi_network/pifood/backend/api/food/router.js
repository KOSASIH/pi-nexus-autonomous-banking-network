const express = require('express');
const router = express.Router();
const FoodController = require('./FoodController');
const authMiddleware = require('../middleware/authMiddleware');
const rateLimitMiddleware = require('../middleware/rateLimitMiddleware');
const validationMiddleware = require('../middleware/validationMiddleware');

const foodController = new FoodController();

router.use(authMiddleware.authenticate);
router.use(rateLimitMiddleware.limit(100, 60 * 60 * 1000)); // 100 requests per hour

router.get('/foods', validationMiddleware.validateQueryParams, foodController.getFoods);
router.get('/foods/:id', validationMiddleware.validateParams, foodController.getFoodById);
router.post('/foods', validationMiddleware.validateBody, foodController.createFood);
router.put('/foods/:id', validationMiddleware.validateParams, validationMiddleware.validateBody, foodController.updateFood);
router.delete('/foods/:id', validationMiddleware.validateParams, foodController.deleteFood);

router.get('/foods/search', validationMiddleware.validateQueryParams, foodController.searchFoods);
router.get('/foods/categories', foodController.getFoodCategories);
router.get('/foods/categories/:category', validationMiddleware.validateParams, foodController.getFoodsByCategory);

router.post('/foods/:id/reviews', validationMiddleware.validateParams, validationMiddleware.validateBody, foodController.createFoodReview);
router.get('/foods/:id/reviews', validationMiddleware.validateParams, foodController.getFoodReviews);

router.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ message: 'Internal Server Error' });
});

module.exports = router;
