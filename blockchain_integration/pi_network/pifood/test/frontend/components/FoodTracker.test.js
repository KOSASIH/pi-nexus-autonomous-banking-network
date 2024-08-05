// FoodTracker.test.js
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { FoodTracker } from './FoodTracker';
import { FoodItem } from '../models/FoodItem';
import { foodData } from '../data/foodData';

describe('FoodTracker component', () => {
  it('renders a list of food items', () => {
    const { getByText } = render(<FoodTracker />);
    foodData.forEach((foodItem) => {
      expect(getByText(foodItem.name)).toBeInTheDocument();
    });
  });

  it('allows user to add a new food item', () => {
    const { getByPlaceholderText, getByText } = render(<FoodTracker />);
    const newFoodItemName = 'New Food Item';
    const newFoodItemDescription = 'This is a new food item';
    const newFoodItemPrice = 10.99;

    const nameInput = getByPlaceholderText('Enter food item name');
    const descriptionInput = getByPlaceholderText('Enter food item description');
    const priceInput = getByPlaceholderText('Enter food item price');

    fireEvent.change(nameInput, { target: { value: newFoodItemName } });
    fireEvent.change(descriptionInput, { target: { value: newFoodItemDescription } });
    fireEvent.change(priceInput, { target: { value: newFoodItemPrice } });

    const addButton = getByText('Add Food Item');
    fireEvent.click(addButton);

    expect(getByText(newFoodItemName)).toBeInTheDocument();
  });

  it('allows user to edit an existing food item', () => {
    const { getByText, getByPlaceholderText } = render(<FoodTracker />);
    const foodItemToEdit = foodData[0];
    const editedFoodItemName = 'Edited Food Item';
    const editedFoodItemDescription = 'This is an edited food item';
    const editedFoodItemPrice = 12.99;

    const editButton = getByText('Edit');
    fireEvent.click(editButton);

    const nameInput = getByPlaceholderText('Enter food item name');
    const descriptionInput = getByPlaceholderText('Enter food item description');
    const priceInput = getByPlaceholderText('Enter food item price');

    fireEvent.change(nameInput, { target: { value: editedFoodItemName } });
    fireEvent.change(descriptionInput, { target: { value: editedFoodItemDescription } });
    fireEvent.change(priceInput, { target: { value: editedFoodItemPrice } });

    const saveButton = getByText('Save');
    fireEvent.click(saveButton);

    expect(getByText(editedFoodItemName)).toBeInTheDocument();
  });

  it('allows user to delete an existing food item', () => {
    const { getByText } = render(<FoodTracker />);
    const foodItemToDelete = foodData[0];

    const deleteButton = getByText('Delete');
    fireEvent.click(deleteButton);

    expect(getByText(foodItemToDelete.name)).not.toBeInTheDocument();
  });

  it('displays a loading message when data is being fetched', async () => {
    const { getByText } = render(<FoodTracker />);
    expect(getByText('Loading...')).toBeInTheDocument();

    await waitFor(() => getByText('Food Items'));

    expect(getByText('Loading...')).not.toBeInTheDocument();
  });
});
