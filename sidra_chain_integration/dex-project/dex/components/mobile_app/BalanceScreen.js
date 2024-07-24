import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button } from 'react-native';
import { connect } from 'react-redux';
import { fetchBalance } from '../actions';

const BalanceScreen = ({ balance, fetchBalance }) => {
  const [address, setAddress] = useState('');

  useEffect(() => {
    fetchBalance(address);
  }, [address]);

  const handleAddressChange = (text) => {
    setAddress(text);
  };

  return (
    <View>
      <Text>Enter your Ethereum address:</Text>
      <TextInput value={address} onChangeText={handleAddressChange} />
      <Button title="Get Balance" onPress={() => fetchBalance(address)} />
      <Text>Balance: {balance}</Text>
    </View>
  );
};

const mapStateToProps = (state) => {
  return {
    balance: state.balance,
  };
};

const mapDispatchToProps = (dispatch) => {
  return bindActionCreators({ fetchBalance }, dispatch);
};

export default connect(mapStateToProps, mapDispatchToProps)(BalanceScreen);
