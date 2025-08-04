// chain_bridge.rs
// A Rust file to implement a bridge between different blockchain networks
//! Cross-chain bridge for interoperability

use hyperbridge::{Chain, Config, Hyperbridge};

pub struct ChainBridge {
    bridge: Hyperbridge,
}

impl ChainBridge {
    /// Initialize a new ChainBridge instance
    pub fn new(config: Config) -> Self {
        let bridge = Hyperbridge::new(config);
        ChainBridge { bridge }
    }

    /// Register a new chain with the bridge
    pub fn register_chain(&mut self, chain: Chain) -> Result<(), &'static str> {
        self.bridge.register_chain(chain)
    }

    /// Transfer assets between chains
    pub fn transfer_assets(
        &mut self,
        from_chain: Chain,
        to_chain: Chain,
        amount: u128,
    ) -> Result<(), &'static str> {
        self.bridge.transfer_assets(from_chain, to_chain, amount)
    }
}
