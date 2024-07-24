import { combineReducers } from 'edux';

const aiReducer = (state = {}, action) => {
  switch (action.type) {
    case 'TRAIN_MODEL_SUCCESS':
      return {...state, modelTrained: true };
    case 'PREDICT_SUCCESS':
      return {...state, prediction: action.payload };
    default:
      return state;
  }
};

export default combineReducers({
  ai: aiReducer,
});
