import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';
import { useWalletConnect } from '../../hooks/useWalletConnect';

const Settings = () => {
  const dispatch = useDispatch();
  const settingsState = useSelector((state) => state.settings);
  const [theme, setTheme] = useState('light');
  const { connectWallet, walletConnected } = useWalletConnect();

  const handleThemeChange = () => {
    if (theme === 'light') {
      setTheme('dark');
    } else {
      setTheme('light');
    }
  };

  return (
    <View style={styles.container}>
      <Text>Settings</Text>
      <View style={styles.settingContainer}>
        <Text>Theme:</Text>
        <TouchableOpacity onPress={handleThemeChange}>
          <Text>{theme}</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.settingContainer}>
        <Text>Wallet:</Text>
        <TouchableOpacity onPress={connectWallet}>
          <Text>{walletConnected ? 'Connected' : 'Connect'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  settingContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
});

export default Settings;
