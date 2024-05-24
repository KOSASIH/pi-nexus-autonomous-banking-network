import React, { useState, useEffect } from 'react';
import { View, Text, FlatList } from 'react-native';
import { API } from '../api';

const Dashboard = () => {
  const [accounts, setAccounts] = useState([]);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    API.getAccounts().then((response) => {
      setAccounts(response.data);
    });
    API.getTransactions().then((response) => {
      setTransactions(response.data);
    });
  }, []);

  return (
    <View>
      <Text>Dashboard</Text>
      <FlatList
        data={accounts}
        renderItem={({ item }) => (
          <View>
            <Text>{item.accountName}</Text>
            <Text>{item.balance}</Text>
          </View>
        )}
      />
      <FlatList
        data={transactions}
        renderItem={({ item }) => (
          <View>
            <Text>{item.transactionDate}</Text>
            <Text>{item.transactionAmount}</Text>
          </View>
        )}
      />
    </View>
  );
};

export default Dashboard;
