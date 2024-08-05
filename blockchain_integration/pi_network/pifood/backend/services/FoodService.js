// FoodService.js
const Food = require('../models/Food');
const Review = require('../models/Review');
const { NotFoundError, BadRequestError } = require('../errors');

class FoodService {
  async createFood(foodData) {
    const food = new Food(foodData);
    await food.save();
    return food;
  }

  async getFoods(filter = {}) {
    const foods = await Food.find(filter).populate('reviews');
    return foods;
  }

  async getFood(id) {
    const food = await Food.findById(id).populate('reviews');
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    return food;
  }

  async updateFood(id, updates) {
    const food = await Food.findByIdAndUpdate(id, updates, { new: true });
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    return food;
  }

  async deleteFood(id) {
    const food = await Food.findByIdAndRemove(id);
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    return food;
  }

  async addReview(foodId, reviewData) {
    const food = await Food.findById(foodId);
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    const review = new Review(reviewData);
    food.reviews.push(review);
    await food.save();
    return review;
  }

  async getReviews(foodId) {
    const food = await Food.findById(foodId).populate('reviews');
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    return food.reviews;
  }

  async getAverageRating(foodId) {
    const food = await Food.findById(foodId).populate('reviews');
    if (!food) {
      throw new NotFoundError('Food not found');
    }
    const sum = food.reviews.reduce((acc, review) => acc + review.rating, 0);
    return sum / food.reviews.length;
  }
}

module.exports = FoodService;
