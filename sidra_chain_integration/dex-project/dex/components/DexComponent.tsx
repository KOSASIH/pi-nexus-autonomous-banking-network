import React, { useState, useEffect } from 'eact';
import { View, Text, StyleSheet, TouchableOpacity } from 'eact-native';
import { useDispatch, useSelector } from 'eact-redux';
import { dexActions } from '../actions';
import { dexReducer } from '../reducers';
import { useWalletConnect } from '../hooks/useWalletConnect';

const DexComponent = () => {
  const dispatch = useDispatch();
  const dexState = useSelector((state) => state.dex);
  const [loading, setLoading] = useState(false);
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(dexActions.fetchDexData());
    }
  }, [walletConnected]);

  const handleTrade = () => {
    // implement trade logic
  };

  return (
    <View style={styles.container}>
      <Text>Dex Component</Text>
      <Text>Dex Data: {dexState.data}</Text>
      <TouchableOpacity onPress={handleTrade}>
        <Text>Trade</Text>
      </TouchableOpacity>
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

export default DexComponent;
