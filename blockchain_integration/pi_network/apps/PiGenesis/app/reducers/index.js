import { combineReducers } from 'edux';
import { reducer as portfolioReducer } from './portfolioReducer';

const rootReducer = combineReducers({
  portfolio: portfolioReducer,
});

export default rootReducer;
