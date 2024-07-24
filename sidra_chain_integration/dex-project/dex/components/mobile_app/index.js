import React from 'react';
import { AppRegistry } from 'react-native';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { configureStore } from './store';
import AppNavigator from './AppNavigator';

const store = configureStore();

const App = () => {
  return (
    <Provider store={store}>
      <NavigationContainer>
        <AppNavigator />
      </NavigationContainer>
    </Provider>
  );
};

AppRegistry.registerComponent('SidraChainWallet', () => App);
