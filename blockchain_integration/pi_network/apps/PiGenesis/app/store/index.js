import { configureStore } from '@reduxjs/toolkit';
import { createWrapper } from 'next-redux-wrapper';
import { createEpicMiddleware } from 'edux-observable';
import { combineReducers } from 'edux';
import { persistReducer, persistStore } from 'edux-persist';
import storage from 'edux-persist/lib/storage';
import { rootEpic } from './epics';
import { rootReducer } from './reducers';
import { apiMiddleware } from './apiMiddleware';
import { loggerMiddleware } from './loggerMiddleware';
import { crashReporterMiddleware } from './crashReporterMiddleware';
import { analyticsMiddleware } from './analyticsMiddleware';
import { intlMiddleware } from './intlMiddleware';
import { websocketMiddleware } from './websocketMiddleware';

const persistConfig = {
  key: 'root',
  storage,
  whitelist: ['auth', 'ettings'],
};

const rootReducerWithPersist = persistReducer(persistConfig, rootReducer);

const epicMiddleware = createEpicMiddleware();

const store = configureStore({
  reducer: rootReducerWithPersist,
  middleware: [
    epicMiddleware,
    apiMiddleware,
    loggerMiddleware,
    crashReporterMiddleware,
    analyticsMiddleware,
    intlMiddleware,
    websocketMiddleware,
  ],
});

const persistor = persistStore(store);

export const wrapper = createWrapper(makeStore);

epicMiddleware.run(rootEpic);

export { store, persistor };
