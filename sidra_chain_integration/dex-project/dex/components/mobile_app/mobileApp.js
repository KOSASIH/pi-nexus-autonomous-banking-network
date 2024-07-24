import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { fetchBalance, fetchTransactionHistory } from '../actions';

const MobileApp = () => {
  const dispatch = useDispatch();
  const balance = useSelector((state) => state.balance);
  const transactionHistory = useSelector((state) => state.transactionHistory);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    dispatch(fetchBalance());
    dispatch(fetchTransactionHistory());
  }, []);

  const handleSendTransaction = () => {
    // implement send transaction logic
  };

  return (
    <View style={styles.container}>
      <Text>Mobile App</Text>
      <Text>Balance: {balance}</Text>
      <Text>Transaction History:</Text>
      <TouchableOpacity onPress={handleSendTransaction}>
        <Text>Send Transaction</Text>
      </TouchableOpacity>
      {transactionHistory.map((transaction) => (
        <Text key={transaction.id}>{transaction.amount} {transaction.currency}</Text>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default MobileApp;
