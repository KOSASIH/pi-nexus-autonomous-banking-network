// multi_chain.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::chain::{Chain, ChainConfig};
use crate::consensus::{ConsensusAlgorithm, ConsensusConfig};
use crate::cross_chain_bridge::{BridgeProtocol, CrossChainBridge};
use crate::wallet::{Wallet, WalletConfig};

pub struct MultiChain {
    chains: HashMap<String, Arc<Chain>>,
    chain_configs: HashMap<String, ChainConfig>,
    consensus_algorithms: HashMap<String, ConsensusAlgorithm>,
    bridge_protocols: HashMap<String, BridgeProtocol>,
    wallets: HashMap<String, Arc<Wallet>>,
    wallet_configs: HashMap<String, WalletConfig>,
    cross_chain_bridges: HashMap<String, Arc<CrossChainBridge>>,
}

impl MultiChain {
    pub fn new() -> Self {
        MultiChain {
            chains: HashMap::new(),
            chain_configs: HashMap::new(),
            consensus_algorithms: HashMap::new(),
            bridge_protocols: HashMap::new(),
            wallets: HashMap::new(),
            wallet_configs: HashMap::new(),
            cross_chain_bridges: HashMap::new(),
        }
    }

    pub fn add_chain(&mut self, chain_id: String, chain_config: ChainConfig) {
        self.chains.insert(chain_id.clone(), Arc::new(Chain::new(chain_config)));
        self.chain_configs.insert(chain_id, chain_config);
    }

    pub fn add_consensus_algorithm(&mut self, algorithm_id: String, algorithm: ConsensusAlgorithm) {
        self.consensus_algorithms.insert(algorithm_id, algorithm);
    }

    pub fn add_bridge_protocol(&mut self, protocol_id: String, protocol: BridgeProtocol) {
        self.bridge_protocols.insert(protocol_id, protocol);
    }

    pub fn add_wallet(&mut self, wallet_id: String, wallet_config: WalletConfig) {
        self.wallets.insert(wallet_id.clone(), Arc::new(Wallet::new(wallet_config)));
        self.wallet_configs.insert(wallet_id, wallet_config);
    }

    pub fn add_cross_chain_bridge(&mut self, bridge_id: String, bridge: CrossChainBridge) {
        self.cross_chain_bridges.insert(bridge_id, Arc::new(bridge));
    }

    pub fn get_chain(&self, chain_id: &str) -> Option<Arc<Chain>> {
        self.chains.get(chain_id).cloned()
    }

    pub fn get_consensus_algorithm(&self, algorithm_id: &str) -> Option<ConsensusAlgorithm> {
        self.consensus_algorithms.get(algorithm_id).cloned()
    }

    pub fn get_bridge_protocol(&self, protocol_id: &str) -> Option<BridgeProtocol> {
        self.bridge_protocols.get(protocol_id).cloned()
    }

    pub fn get_wallet(&self, wallet_id: &str) -> Option<Arc<Wallet>> {
        self.wallets.get(wallet_id).cloned()
    }

    pub fn get_cross_chain_bridge(&self, bridge_id: &str) -> Option<Arc<CrossChainBridge>> {
        self.cross_chain_bridges.get(bridge_id).cloned()
    }
}
