import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity } from "react-native";
import { BiometricAuth } from "expo-biometrics";
import { API } from "../api";

const Login = ({ navigation }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [biometricAuth, setBiometricAuth] = useState(false);

  const handleLogin = async () => {
    if (biometricAuth) {
      const result = await BiometricAuth.authenticateAsync();
      if (result.success) {
        const token = result.token;
        API.authenticate(token)
          .then((response) => {
            navigation.navigate("Dashboard");
          })
          .catch((error) => {
            console.error(error);
          });
      } else {
        console.error("Biometric authentication failed");
      }
    } else {
      API.authenticate(username, password)
        .then((response) => {
          navigation.navigate("Dashboard");
        })
        .catch((error) => {
          console.error(error);
        });
    }
  };

  return (
    <View>
      <Text>Login</Text>
      <TextInput
        placeholder="Username"
        value={username}
        onChangeText={(text) => setUsername(text)}
      />
      <TextInput
        placeholder="Password"
        value={password}
        onChangeText={(text) => setPassword(text)}
        secureTextEntry
      />
      {biometricAuth ? (
        <TouchableOpacity onPress={handleLogin}>
          <Text>Use Biometric Authentication</Text>
        </TouchableOpacity>
      ) : (
        <TouchableOpacity onPress={handleLogin}>
          <Text>Login</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

export default Login;
