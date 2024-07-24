#import <React/RCTBridgeModule.h>
#import <WalletConnect/WalletConnect.h>

@interface MobileAppNativeModule : NSObject <RCTBridgeModule>

@end

@implementation MobileAppNativeModule

RCT_EXPORT_MODULE();

- (dispatch_async)connectWallet {
  [[WalletConnect sharedInstance] connect];
}

- (dispatch_async)getBalance {
  [[WalletConnect sharedInstance] getBalance:^(NSString *balance, NSError *error) {
    if (error) {
      NSLog(@"%@", error);
    } else {
      resolve(balance);
    }
  }];
}

- (dispatch_async)getTransactionHistory {
  [[WalletConnect sharedInstance] getTransactionHistory:^(NSArray<NSString *> *transactionHistory, NSError *error) {
    if (error) {
      NSLog(@"%@", error);
    } else {
      resolve(transactionHistory);
    }
  }];
}

- (dispatch_async)sendTransaction:(NSString *)amount recipient:(NSString *)recipient {
  [[WalletConnect sharedInstance] sendTransaction:amount recipient:recipient:^(NSError *error) {
    if (error) {
      NSLog(@"%@", error);
    } else {
      resolve();
    }
  }];
}

@end
