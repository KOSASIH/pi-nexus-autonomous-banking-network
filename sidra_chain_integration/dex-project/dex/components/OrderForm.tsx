import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, Button } from 'react-native';
import { useWalletConnect } from '../../hooks/useWalletConnect';
import { useWebSocket } from '../../hooks/useWebSocket';

const OrderForm = () => {
  const { walletConnected, walletAddress } = useWalletConnect();
  const { sendMessage } = useWebSocket();
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const [side, setSide] = useState('');

  const handleSubmit = () => {
    if (walletConnected && amount && price && side) {
      const order = {
        type: 'limit_order',
        amount,
        price,
        side,
        walletAddress,
      };
      sendMessage(JSON.stringify(order));
    }
  };

  return (
    <View style={styles.container}>
      <Text>Order Form</Text>
      {walletConnected ? (
        <View>
          <TextInput
            style={styles.input}
            placeholder="Amount"
            value={amount}
            onChangeText={(text) => setAmount(text)}
          />
          <TextInput
            style={styles.input}
            placeholder="Price"
            value={price}
            onChangeText={(text) => setPrice(text)}
          />
          <TextInput
            style={styles.input}
            placeholder="Side (buy/sell)"
            value={side}
            onChangeText={(text) => setSide(text)}
          />
          <Button title="Submit Order" onPress={handleSubmit} />
        </View>
      ) : (
        <Text>Please connect to a wallet</Text>
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
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    margin: 10,
    padding: 10,
  },
});

export default OrderForm;
