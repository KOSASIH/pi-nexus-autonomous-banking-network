// InventoryManager.js
import React, { useState, useEffect } from 'react';
import { FoodService } from '../services/FoodService';
import { UserService } from '../services/UserService';

const InventoryManager = () => {
  const [foods, setFoods] = useState([]);
  const [selectedFood, setSelectedFood] = useState(null);
  const [quantity, setQuantity] = useState(0);
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

  const handleSelectFood = (food) => {
    setSelectedFood(food);
  };

  const handleUpdateQuantity = async (event) => {
    event.preventDefault();
    try {
      await foodService.updateFood(selectedFood.id, { quantity });
      setQuantity(0);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  return (
    <div>
      <h1>Inventory Manager</h1>
      <ul>
        {foods.map((food) => (
          <li key={food.id}>
            {food.name}
            <button onClick={() => handleSelectFood(food)}>Select</button>
          </li>
        ))}
      </ul>
      {selectedFood && (
        <form onSubmit={handleUpdateQuantity}>
          <label>
            Quantity:
            <input
              type="number"
              value={quantity}
              onChange={(event) => setQuantity(event.target.value)}
            />
          </label>
          <button type="submit">Update</button>
        </form>
      )}
      {errorMessage && <div style={{ color: 'red' }}>{errorMessage}</div>}
    </div>
  );
};

export default InventoryManager;
