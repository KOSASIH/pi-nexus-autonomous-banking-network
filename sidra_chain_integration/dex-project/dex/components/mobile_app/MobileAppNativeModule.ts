import { NativeModules } from 'react-native';
import { WalletConnect } from '@walletconnect/client';

const { MobileAppNativeModule } = NativeModules;

interface MobileAppNativeModule {
  connectWallet(): Promise<void>;
  getBalance(): Promise<string>;
  getTransactionHistory(): Promise<string[]>;
  sendTransaction(amount: string, recipient: string): Promise<void>;
}

const mobileAppNativeModule = new MobileAppNativeModule();

export default mobileAppNativeModule;
