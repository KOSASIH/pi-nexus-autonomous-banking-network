import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import BalanceScreen from '../screens/BalanceScreen';
import TransactionHistoryScreen from '../screens/TransactionHistoryScreen';

const Stack = createStackNavigator();

const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Balance" component={BalanceScreen} />
        <Stack.Screen name="Transaction History" component={TransactionHistoryScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;
