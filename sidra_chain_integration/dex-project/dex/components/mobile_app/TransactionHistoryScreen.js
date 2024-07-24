import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, FlatList } from 'react-native';
import { connect } from 'react-redux';
import { fetchTransactionHistory } from '../actions';

const TransactionHistoryScreen = ({ transactionHistory, fetchTransactionHistory }) => {
  const [address, setAddress] = useState('');

  useEffect(() => {
    fetchTransactionHistory(address);
  }, [address]);

  const handleAddressChange = (text) => {
    setAddress(text);
  };

  return (
    <View>
      <Text>Enter your Ethereum address:</Text>
      <TextInput value={address} onChangeText={handleAddressChange} />
      <Button title="Get Transaction History" onPress={() => fetchTransactionHistory(address)} />
      <FlatList
        data={transactionHistory}
        renderItem={({ item }) => (
          <View>
            <Text>Transaction Hash: {item.transactionHash}</Text>
            <Text>Block Number: {item.blockNumber}</Text>
            <Text>From: {item.from}</Text>
            <Text>To: {item.to}</Text>
            <Text>Value: {item.value}</Text>
          </View>
        )}
      />
    </View>
  );
};

const mapStateToProps = (state) => {
  return {
    transactionHistory: state.transactionHistory,
  };
};

const mapDispatchToProps = (dispatch) => {
  return bindActionCreators({ fetchTransactionHistory }, dispatch);
};

export default connect(mapStateToProps, mapDispatchToProps)(TransactionHistoryScreen);
