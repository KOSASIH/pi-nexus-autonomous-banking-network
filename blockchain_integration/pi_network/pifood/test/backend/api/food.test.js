// food.test.js
import supertest from 'supertest';
import app from '../app';
import { FOOD_API_ENDPOINTS } from '../constants/apiEndpoints';
import { FOOD_DATA } from '../data/foodData';

const request = supertest(app);

describe('Food API', () => {
  describe('GET /food', () => {
    it('should return a list of food items', async () => {
      const response = await request.get(FOOD_API_ENDPOINTS.GET_FOOD);
      expect(response.status).toBe(200);
      expect(response.body).toBeInstanceOf(Array);
      expect(response.body.length).toBeGreaterThan(0);
    });
  });

  describe('GET /food/:id', () => {
    it('should return a single food item by ID', async () => {
      const foodId = FOOD_DATA[0].id;
      const response = await request.get(`${FOOD_API_ENDPOINTS.GET_FOOD}/${foodId}`);
      expect(response.status).toBe(200);
      expect(response.body).toBeInstanceOf(Object);
      expect(response.body.id).toBe(foodId);
    });
  });

  describe('POST /food', () => {
    it('should create a new food item', async () => {
      const newFoodItem = {
        name: 'New Food Item',
        description: 'This is a new food item',
        price: 10.99,
      };
      const response = await request.post(FOOD_API_ENDPOINTS.CREATE_FOOD).send(newFoodItem);
      expect(response.status).toBe(201);
      expect(response.body).toBeInstanceOf(Object);
      expect(response.body.name).toBe(newFoodItem.name);
    });
  });

  describe('PUT /food/:id', () => {
    it('should update an existing food item', async () => {
      const foodId = FOOD_DATA[0].id;
      const updatedFoodItem = {
        name: 'Updated Food Item',
        description: 'This is an updated food item',
        price: 12.99,
      };
      const response = await request.put(`${FOOD_API_ENDPOINTS.UPDATE_FOOD}/${foodId}`).send(updatedFoodItem);
      expect(response.status).toBe(200);
      expect(response.body).toBeInstanceOf(Object);
      expect(response.body.name).toBe(updatedFoodItem.name);
    });
  });

  describe('DELETE /food/:id', () => {
    it('should delete an existing food item', async () => {
      const foodId = FOOD_DATA[0].id;
      const response = await request.delete(`${FOOD_API_ENDPOINTS.DELETE_FOOD}/${foodId}`);
      expect(response.status).toBe(204);
    });
  });
});
