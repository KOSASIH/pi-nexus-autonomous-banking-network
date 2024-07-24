import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, TextInput } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { dexActions } from '../../actions';
import { dexReducer } from '../../reducers';

const Settings = () => {
  const dispatch = useDispatch();
  const settingsState = useSelector((state) => state.settings);
  const [darkMode, setDarkMode] = useState(settingsState.darkMode);
  const [apiUrl, setApiUrl] = useState(settingsState.apiUrl);

  const handleDarkModeChange = (value) => {
    setDarkMode(value);
    dispatch(dexActions.updateSettings({ darkMode: value }));
  };

  const handleApiUrlChange = (text) => {
    setApiUrl(text);
    dispatch(dexActions.updateSettings({ apiUrl: text }));
  };

  return (
    <View style={styles.container}>
      <Text>Settings</Text>
      <View style={styles.settingContainer}>
        <Text>Dark Mode</Text>
        <Switch value={darkMode} onValueChange={handleDarkModeChange} />
      </View>
      <View style={styles.settingContainer}>
        <Text>API URL</Text>
        <TextInput
          style={styles.input}
          value={apiUrl}
          onChangeText={handleApiUrlChange}
        />
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
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    padding: 10,
    margin: 10,
  },
});

export default Settings;
