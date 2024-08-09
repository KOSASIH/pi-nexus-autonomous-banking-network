import { configureStore } from '@reduxjs/toolkit';
import { authReducer } from '../reducers/auth.reducers';
import { rootReducer } from '../reducers';

const store = configureStore({
  reducer: {
    auth: authReducer,
    root: rootReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
