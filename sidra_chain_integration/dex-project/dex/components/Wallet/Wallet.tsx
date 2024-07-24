import React, { useState, useEffect } from 'eact';
import { View, Text, StyleSheet, Button } from 'eact-native';
import { useDispatch, useSelector } from 'eact-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const Wallet = () => {
  const dispatch = useDispatch();
  const walletState = useSelector((state) => state.wallet);
  const [walletAddress, setWalletAddress] = useState('');
  const { connectWallet, walletConnected } = useWalletConnect();

  useEffect(() => {
    if (walletConnected) {
      dispatch(dexActions.fetchWalletBalance());
    }
  }, [walletConnected]);

  const handleConnectWallet = () => {
    connectWallet();
  };

  return (
    <View style={styles.container}>
      <Text>Wallet</Text>
      {walletConnected? (
        <Text>Connected to {walletAddress}</Text>
      ) : (
        <Button title="Connect Wallet" onPress={handleConnectWallet} />
      )}
      <Text>Balance: {walletState.balance}</Text>
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

export default Wallet;
