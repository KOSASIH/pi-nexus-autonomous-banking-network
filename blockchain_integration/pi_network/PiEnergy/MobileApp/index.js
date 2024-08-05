import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, FlatList } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import * as Permissions from 'expo-permissions';
import { Camera, CameraConstants } from 'expo-camera';
import { Audio } from 'expo-av';

const Stack = createStackNavigator();

const App = () => {
  const [location, setLocation] = useState(null);
  const [cameraPermission, setCameraPermission] = useState(null);
  const [audioPermission, setAudioPermission] = useState(null);
  const [images, setImages] = useState([]);
  const [audioRecordings, setAudioRecordings] = useState([]);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestPermissionsAsync();
      if (status !== 'granted') {
        console.error('Location permission denied');
      } else {
        const location = await Location.getCurrentPositionAsync();
        setLocation(location);
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const { status } = await Permissions.askAsync(Permissions.CAMERA);
      setCameraPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const { status } = await Permissions.askAsync(Permissions.AUDIO_RECORDING);
      setAudioPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (cameraPermission) {
      const camera = new Camera();
      const photo = await camera.takePictureAsync();
      setImages([...images, photo.uri]);
    }
  };

  const recordAudio = async () => {
    if (audioPermission) {
      const audio = new Audio.Recording();
      await audio.prepareToRecordAsync();
      await audio.startAsync();
      const recording = await audio.stopAsync();
      setAudioRecordings([...audioRecordings, recording.uri]);
    }
  };

  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="Audio" component={AudioScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const HomeScreen = () => {
  return (
    <View>
      <Text>Welcome to the Mobile App!</Text>
      <TouchableOpacity onPress={takePicture}>
        <Text>Take a Picture</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={recordAudio}>
        <Text>Record Audio</Text>
      </TouchableOpacity>
    </View>
  );
};

const CameraScreen = () => {
  return (
    <View>
      <Camera style={{ flex: 1 }} type={Camera.Constants.Type.back}>
        <View>
          <MaterialCommunityIcons name="camera" size={30} color="white" />
        </View>
      </Camera>
    </View>
  );
};

const AudioScreen = () => {
  return (
    <View>
      <FlatList
        data={audioRecordings}
        renderItem={({ item }) => (
          <View>
            <Text>{item}</Text>
          </View>
        )}
      />
    </View>
  );
};

export default App;
