// Food.js
const mongoose = require('mongoose');
const { Schema } = mongoose;
const reviewSchema = require('./Review');

const foodSchema = new Schema({
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
  price: {
    type: Number,
    required: true,
    min: 0,
  },
  category: {
    type: String,
    required: true,
    enum: ['breakfast', 'lunch', 'dinner', 'snack'],
  },
  image: {
    type: String,
    required: true,
  },
  reviews: [reviewSchema],
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

foodSchema.methods.toJSON = function () {
  const food = this;
  const foodObject = food.toObject();
  delete foodObject.reviews;
  return foodObject;
};

foodSchema.virtual('averageRating').get(function () {
  if (!this.reviews || this.reviews.length === 0) {
    return 0;
  }
  const sum = this.reviews.reduce((acc, review) => acc + review.rating, 0);
  return sum / this.reviews.length;
});

const Food = mongoose.model('Food', foodSchema);

module.exports = Food;
