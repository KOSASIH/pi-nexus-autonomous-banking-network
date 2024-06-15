// asset_manager.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::chain::{Chain, ChainConfig};
use crate::wallet::{Wallet, WalletConfig};

pub struct AssetManager {
    assets: HashMap<String, Asset>,
    asset_registry: HashMap<String, AssetRegistry>,
}

impl AssetManager {
    pub fn new() -> Self {
        AssetManager {
            assets: HashMap::new(),
            asset_registry: HashMap::new(),
        }
    }

    pub fn register_asset(&mut self, asset_id: String, asset: Asset) {
        self.assets.insert(asset_id.clone(), asset);
        self.asset_registry.insert(asset_id, AssetRegistry::new(asset));
    }

    pub fn get_asset(&self, asset_id: &str) -> Option<&Asset> {
        self.assets.get(asset_id)
    }

    pub fn get_asset_registry(&self, asset_id: &str) -> Option<&AssetRegistry> {
        self.asset_registry.get(asset_id)
    }
}

pub struct Asset {
    asset_id: String,
    asset_type: AssetType,
    asset_data: Vec<u8>,
}

impl Asset {
    pub fn new(asset_id: String, asset_type: AssetType, asset_data: Vec<u8>) -> Self {
        Asset {
            asset_id,
            asset_type,
            asset_data,
        }
    }

    pub fn get_asset_id(&self) -> &str {
        &self.asset_id
    }

    pub fn get_asset_type(&self) -> &AssetType {
        &self.asset_type
    }

    pub fn get_asset_data(&self) -> &[u8] {
        &self.asset_data
    }
}

pub enum AssetType {
    Token,
    Coin,
    NFT,
}

pub struct AssetRegistry {
    asset_id: String,
    asset_type: AssetType,
    asset_data: Vec<u8>,
}

impl AssetRegistry {
    pub fn new(asset: Asset) -> Self {
        AssetRegistry {
            asset_id: asset.asset_id.clone(),
            asset_type: asset.asset_type,
            asset_data: asset.asset_data.clone(),
        }
    }

    pub fn get_asset_id(&self) -> &str {
        &self.asset_id
    }

    pub fn get_asset_type(&self) -> &AssetType {
        &self.asset_type
    }

    pub fn get_asset_data(&self) -> &[u8] {
        &self.asset_data
    }
}
