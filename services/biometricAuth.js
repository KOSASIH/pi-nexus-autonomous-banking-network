import { Platform } from 'react-native';
import { LocalAuthentication } from 'expo';

export const authenticateWithBiometrics = async () => {
  if (Platform.OS === 'ios') {
    const isAvailable = await LocalAuthentication.hasHardwareAsync();
    if (!isAvailable) return false;
    const isEnrolled = await LocalAuthentication.isEnrolledAsync();
    if (!isEnrolled) return false;
    const result = await LocalAuthentication.authenticateAsync();
    return result.success;
  } else if (Platform.OS === 'android') {
    const isAvailable = await LocalAuthentication.hasHardwareAsync();
    if (!isAvailable) return false;
    const result = await LocalAuthentication.authenticateAsync();
    return result.success;
  }
  return false;
};
