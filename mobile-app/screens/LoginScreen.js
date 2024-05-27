import React from "react";
import { View, Text, Image } from "react-native";
import { Login } from "../components";

const LoginScreen = ({ navigation }) => {
  return (
    <View>
      <Image source={require("../images/fingerprint.png")} />
      <Text>Welcome to the most advanced high-tech mobile app!</Text>
      <Login navigation={navigation} />
    </View>
  );
};

export default LoginScreen;
