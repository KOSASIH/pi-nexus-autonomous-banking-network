// portfolio.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::asset::{Asset, AssetType};
use crate::wallet::{Wallet, WalletConfig};

pub struct Portfolio {
    portfolio_id: String,
    wallet: Arc<Wallet>,
    assets: HashMap<String, Asset>,
}

impl Portfolio {
    pub fn new(portfolio_id: String, wallet: Arc<Wallet>) -> Self {
        Portfolio {
            portfolio_id,
            wallet,
            assets: HashMap::new(),
        }
    }

    pub fn add_asset(&mut self, asset_id: String, asset: Asset) {
        self.assets.insert(asset_id, asset);
    }

    pub fn remove_asset(&mut self, asset_id: &str) {
        self.assets.remove(asset_id);
    }

    pub fn get_assets(&self) -> &HashMap<String, Asset> {
        &self.assets
    }

    pub fn get_wallet(&self) -> &Arc<Wallet> {
        &self.wallet
    }
}
