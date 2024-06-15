// api.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::multi_chain::{MultiChain, ChainConfig};
use crate::cross_chain_bridge::{CrossChainBridge, BridgeProtocol};
use crate::wallet::{Wallet, WalletConfig};

pub struct API {
    multi_chain: Arc<MultiChain>,
    cross_chain_bridges: HashMap<String, Arc<CrossChainBridge>>,
    wallets: HashMap<String, Arc<Wallet>>,
}

impl API {
    pub fn new(multi_chain: Arc<MultiChain>) -> Self {
        API {
            multi_chain,
            cross_chain_bridges: HashMap::new(),
            wallets: HashMap::new(),
        }
    }

    pub fn add_cross_chain_bridge(&mut self, bridge_id: String, bridge: CrossChainBridge) {
        self.cross_chain_bridges.insert(bridge_id, Arc::new(bridge));
    }

    pub fn add_wallet(&mut self, wallet_id: String, wallet: Wallet) {
        self.wallets.insert(wallet_id, Arc::new(wallet));
    }

    pub fn get_chain(&self, chain_id: &str) -> Option<Arc<Chain>> {
        self.multi_chain.get_chain(chain_id)
    }

    pub fn get_cross_chain_bridge(&self, bridge_id: &str) -> Option<Arc<CrossChainBridge>> {
        self.cross_chain_bridges.get(bridge_id).cloned()
    }

    pub fn get_wallet(&self, wallet_id: &str) -> Option<Arc<Wallet>> {
        self.wallets.get(wallet_id).cloned()
    }
}
