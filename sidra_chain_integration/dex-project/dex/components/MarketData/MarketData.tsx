import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const MarketData = () => {
  const dispatch = useDispatch();
  const marketDataState = useSelector((state) => state.marketData);
  const [loading, setLoading] = useState(false);
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(dexActions.fetchMarketData());
    }
  }, [walletConnected]);

  const renderItem = ({ item }) => {
    return (
      <View style={styles.itemContainer}>
        <Text style={styles.itemText}>{item.symbol}</Text>
        <Text style={styles.itemText}>{item.price}</Text>
        <Text style={styles.itemText}>{item.volume}</Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <Text>Market Data</Text>
      {loading ? (
        <Text>Loading...</Text>
      ) : (
        <FlatList
          data={marketDataState.data}
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

export default MarketData;
