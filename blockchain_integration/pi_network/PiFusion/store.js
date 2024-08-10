import { configureStore } from '@reduxjs/toolkit';
import { createLogger } from 'redux-logger';
import thunkMiddleware from 'redux-thunk';
import { nodeReducer } from './reducers/nodeReducer';
import { authReducer } from './reducers/authReducer';
import { dataReducer } from './reducers/dataReducer';

const logger = createLogger({
  level: 'info',
  collapsed: true,
});

const store = configureStore({
  reducer: {
    nodes: nodeReducer,
    auth: authReducer,
    data: dataReducer,
  },
  middleware: [thunkMiddleware, logger],
  devTools: process.env.NODE_ENV !== 'production',
});

export default store;
