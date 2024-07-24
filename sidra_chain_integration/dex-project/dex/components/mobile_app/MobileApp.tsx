import React, { useState, useEffect } from 'eact';
import { View, Text, StyleSheet, TouchableOpacity } from 'eact-native';
import { useDispatch, useSelector } from 'eact-redux';
import { fetchBalance, fetchTransactionHistory } from '../actions';
import { useWalletConnect } from '../hooks/useWalletConnect';

const MobileApp = () => {
  const dispatch = useDispatch();
  const balance = useSelector((state) => state.balance);
  const transactionHistory = useSelector((state) => state.transactionHistory);
  const [loading, setLoading] = useState(false);
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(fetchBalance());
      dispatch(fetchTransactionHistory());
    }
  }, [walletConnected]);

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
      {!walletConnected && (
        <TouchableOpacity onPress={connectWallet}>
          <Text>Connect Wallet</Text>
        </TouchableOpacity>
      )}
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
