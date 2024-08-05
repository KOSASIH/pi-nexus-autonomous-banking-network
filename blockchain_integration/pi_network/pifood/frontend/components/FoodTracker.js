// FoodTracker.js
import React, { useState, useEffect } from 'react';
import { FoodService } from '../services/FoodService';
import { UserService } from '../services/UserService';

const FoodTracker = () => {
  const [foods, setFoods] = useState([]);
  const [newFood, setNewFood] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const foodService = new FoodService();
  const userService = new UserService();

  useEffect(() => {
    const fetchFoods = async () => {
      try {
        const foods = await foodService.getFoods();
        setFoods(foods);
      } catch (err) {
        setErrorMessage(err.message);
      }
    };
    fetchFoods();
  }, []);

  const handleAddFood = async (event) => {
    event.preventDefault();
    try {
      const food = await foodService.createFood({ name: newFood });
      setFoods([...foods, food]);
      setNewFood('');
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  const handleDeleteFood = async (id) => {
    try {
      await foodService.deleteFood(id);
      setFoods(foods.filter((food) => food.id !== id));
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  return (
    <div>
      <h1>Food Tracker</h1>
      <form onSubmit={handleAddFood}>
        <input
          type="text"
          value={newFood}
          onChange={(event) => setNewFood(event.target.value)}
          placeholder="Add new food"
        />
        <button type="submit">Add</button>
      </form>
      <ul>
        {foods.map((food) => (
          <li key={food.id}>
            {food.name}
            <button onClick={() => handleDeleteFood(food.id)}>Delete</button>
          </li>
        ))}
      </ul>
      {errorMessage && <div style={{ color: 'red' }}>{errorMessage}</div>}
    </div>
  );
};

export default FoodTracker;
