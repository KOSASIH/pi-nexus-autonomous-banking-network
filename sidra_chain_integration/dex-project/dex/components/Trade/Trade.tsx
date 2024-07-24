import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const Trade = () => {
  const dispatch = useDispatch();
  const tradeState = useSelector((state) => state.trade);
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const { connectWallet, walletConnected } = useWalletConnect();

  const handleTrade = () => {
    if (walletConnected) {
      dispatch(dexActions.trade(amount, price));
    }
  };

  return (
    <View style={styles.container}>
      <Text>Trade</Text>
      <View style={styles.inputContainer}>
        <Text>Amount:</Text>
        <TextInput
          value={amount}
          onChangeText={(text) => setAmount(text)}
          style={styles.input}
        />
      </View>
      <View style={styles.inputContainer}>
        <Text>Price:</Text>
        <TextInput
          value={price}
          onChangeText={(text) => setPrice(text)}
          style={styles.input}
        />
      </View>
      <TouchableOpacity onPress={handleTrade}>
        <Text>Trade</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  inputContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  input: {
    fontSize: 16,
    color: '#333',
    padding: 8,
  },
});

export default Trade;
