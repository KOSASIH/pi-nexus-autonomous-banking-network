// wallet.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::chain::{Chain, ChainConfig};
use crate::asset::{Asset, AssetType};

pub struct WalletManager {
    wallets: HashMap<String, Arc<Wallet>>,
    wallet_configs: HashMap<String, WalletConfig>,
}

impl WalletManager {
    pub fn new() -> Self {
        WalletManager {
            wallets: HashMap::new(),
            wallet_configs: HashMap::new(),
        }
    }

    pub fn add_wallet(&mut self, wallet_id: String, wallet_config: WalletConfig) {
        self.wallets.insert(wallet_id.clone(), Arc::new(Wallet::new(wallet_config)));
        self.wallet_configs.insert(wallet_id, wallet_config);
    }

    pub fn get_wallet(&self, wallet_id: &str) -> Option<Arc<Wallet>> {
        self.wallets.get(wallet_id).cloned()
    }

    pub fn get_wallet_config(&self, wallet_id: &str) -> Option<&WalletConfig> {
        self.wallet_configs.get(wallet_id)
    }
}

pub struct Wallet {
    wallet_id: String,
    wallet_config: WalletConfig,
    assets: HashMap<String, Asset>,
}

impl Wallet {
    pub fn new(wallet_config: WalletConfig) -> Self {
        Wallet {
            wallet_id: wallet_config.wallet_id.clone(),
            wallet_config,
            assets: HashMap::new(),
        }
    }

    pub fn get_wallet_id(&self) -> &str {
        &self.wallet_id
    }

    pub fn get_wallet_config(&self) -> &WalletConfig {
        &self.wallet_config
    }

    pub fn get_assets(&self) -> &HashMap<String, Asset> {
        &self.assets
    }

    pub fn add_asset(&mut self, asset_id: String, asset: Asset) {
        self.assets.insert(asset_id, asset);
    }

    pub fn remove_asset(&mut self, asset_id: &str) {
        self.assets.remove(asset_id);
    }
}

pub struct WalletConfig {
    wallet_id: String,
    chain_id: String,
    asset_type: AssetType,
}

implWalletConfig {
    pub fn new(wallet_id: String, chain_id: String, asset_type: AssetType) -> Self {
        WalletConfig {
            wallet_id,
            chain_id,
            asset_type,
        }
    }

    pub fn get_wallet_id(&self) -> &str {
        &self.wallet_id
    }

    pub fn get_chain_id(&self) -> &str {
        &self.chain_id
    }

    pub fn get_asset_type(&self) -> &AssetType {
        &self.asset_type
    }
}
