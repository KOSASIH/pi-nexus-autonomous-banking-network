// foodActions.js
import axios from 'axios';
import { createAction } from 'redux-actions';
import { FOOD_API_URL } from '../constants/api';
import { FOOD_TYPES } from '../constants/foodTypes';
import { notification } from 'antd';

// Action creators
export const fetchFoodsRequest = createAction('FETCH_FOODS_REQUEST');
export const fetchFoodsSuccess = createAction('FETCH_FOODS_SUCCESS');
export const fetchFoodsFailure = createAction('FETCH_FOODS_FAILURE');

export const createFoodRequest = createAction('CREATE_FOOD_REQUEST');
export const createFoodSuccess = createAction('CREATE_FOOD_SUCCESS');
export const createFoodFailure = createAction('CREATE_FOOD_FAILURE');

export const updateFoodRequest = createAction('UPDATE_FOOD_REQUEST');
export const updateFoodSuccess = createAction('UPDATE_FOOD_SUCCESS');
export const updateFoodFailure = createAction('UPDATE_FOOD_FAILURE');

export const deleteFoodRequest = createAction('DELETE_FOOD_REQUEST');
export const deleteFoodSuccess = createAction('DELETE_FOOD_SUCCESS');
export const deleteFoodFailure = createAction('DELETE_FOOD_FAILURE');

// Thunk actions
export const fetchFoods = () => async (dispatch) => {
  dispatch(fetchFoodsRequest());
  try {
    const response = await axios.get(`${FOOD_API_URL}/foods`);
    dispatch(fetchFoodsSuccess(response.data));
  } catch (error) {
    dispatch(fetchFoodsFailure(error));
    notification.error({
      message: 'Error fetching foods',
      description: error.message,
    });
  }
};

export const createFood = (food) => async (dispatch) => {
  dispatch(createFoodRequest());
  try {
    const response = await axios.post(`${FOOD_API_URL}/foods`, food);
    dispatch(createFoodSuccess(response.data));
    notification.success({
      message: 'Food created successfully',
    });
  } catch (error) {
    dispatch(createFoodFailure(error));
    notification.error({
      message: 'Error creating food',
      description: error.message,
    });
  }
};

export const updateFood = (food) => async (dispatch) => {
  dispatch(updateFoodRequest());
  try {
    const response = await axios.put(`${FOOD_API_URL}/foods/${food.id}`, food);
    dispatch(updateFoodSuccess(response.data));
    notification.success({
      message: 'Food updated successfully',
    });
  } catch (error) {
    dispatch(updateFoodFailure(error));
    notification.error({
      message: 'Error updating food',
      description: error.message,
    });
  }
};

export const deleteFood = (id) => async (dispatch) => {
  dispatch(deleteFoodRequest());
  try {
    await axios.delete(`${FOOD_API_URL}/foods/${id}`);
    dispatch(deleteFoodSuccess(id));
    notification.success({
      message: 'Food deleted successfully',
    });
  } catch (error) {
    dispatch(deleteFoodFailure(error));
    notification.error({
      message: 'Error deleting food',
      description: error.message,
    });
  }
};
