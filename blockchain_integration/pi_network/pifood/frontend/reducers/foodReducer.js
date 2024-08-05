// foodReducer.js
import { handleActions } from 'redux-actions';
import { FOOD_TYPES } from '../constants/foodTypes';
import { initialState } from './initialState';

const foodReducer = handleActions(
  {
    FETCH_FOODS_SUCCESS: (state, action) => {
      return { ...state, foods: action.payload };
    },
    FETCH_FOODS_FAILURE: (state, action) => {
      return { ...state, error: action.payload };
    },
    CREATE_FOOD_SUCCESS: (state, action) => {
      return { ...state, foods: [...state.foods, action.payload] };
    },
    CREATE_FOOD_FAILURE: (state, action) => {
      return { ...state, error: action.payload };
    },
    UPDATE_FOOD_SUCCESS: (state, action) => {
      const updatedFood = action.payload;
      const foods = state.foods.map((food) => (food.id === updatedFood.id ? updatedFood : food));
      return { ...state, foods };
    },
    UPDATE_FOOD_FAILURE: (state, action) => {
      return { ...state, error: action.payload };
    },
    DELETE_FOOD_SUCCESS: (state, action) => {
      const id = action.payload;
      const foods = state.foods.filter((food) => food.id !== id);
      return { ...state, foods };
    },
    DELETE_FOOD_FAILURE: (state, action) => {
      return { ...state, error: action.payload };
    },
  },
  initialState.foods
);

export default foodReducer;
