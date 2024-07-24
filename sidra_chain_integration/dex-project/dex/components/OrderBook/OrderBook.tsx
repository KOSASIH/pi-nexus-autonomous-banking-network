import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const OrderBook = () => {
  const dispatch = useDispatch();
  const orderBookState = useSelector((state) => state.orderBook);
  const [loading, setLoading] = useState(false);
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(dexActions.fetchOrderBook());
    }
  }, [walletConnected]);

  const renderItem = ({ item }) => {
    return (
      <View style={styles.itemContainer}>
        <Text style={styles.itemText}>{item.price}</Text>
        <Text style={styles.itemText}>{item.amount}</Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <Text>Order Book</Text>
      {loading ? (
        <Text>Loading...</Text>
      ) : (
        <FlatList
          data={orderBookState.data}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
        />
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
  itemContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  itemText: {
    fontSize: 16,
    color: '#333',
  },
});

export default OrderBook;
