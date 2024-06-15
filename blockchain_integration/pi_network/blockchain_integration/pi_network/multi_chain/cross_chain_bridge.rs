// cross_chain_bridge.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::chain::{Chain, ChainConfig};
use crate::consensus::{ConsensusAlgorithm, ConsensusConfig};
use crate::wallet::{Wallet, WalletConfig};

pub struct CrossChainBridge {
    bridge_id: String,
    chain_id: String,
    wallet_id: String,
    bridge_protocol: BridgeProtocol,
    chain_config: ChainConfig,
    wallet_config: WalletConfig,
    consensus_algorithm: ConsensusAlgorithm,
    cross_chain_messages: VecDeque<CrossChainMessage>,
}

impl CrossChainBridge {
    pub fn new(
        bridge_id: String,
        chain_id: String,
        wallet_id: String,
        bridge_protocol: BridgeProtocol,
        chain_config: ChainConfig,
        wallet_config: WalletConfig,
        consensus_algorithm: ConsensusAlgorithm,
    ) -> Self {
        CrossChainBridge {
            bridge_id,
            chain_id,
            wallet_id,
            bridge_protocol,
            chain_config,
            wallet_config,
            consensus_algorithm,
            cross_chain_messages: VecDeque::new(),
        }
    }

    pub fn send_cross_chain_message(&mut self, message: CrossChainMessage) {
        self.cross_chain_messages.push_back(message);
    }

    pub fn receive_cross_chain_message(&mut self) -> Option<CrossChainMessage> {
        self.cross_chain_messages.pop_front()
    }

    pub fn get_bridge_id(&self) -> &str {
        &self.bridge_id
    }

    pub fn get_chain_id(&self) -> &str {
        &self.chain_id
    }

    pub fn get_wallet_id(&self) -> &str {
        &self.wallet_id
    }

    pub fn get_bridge_protocol(&self) -> &BridgeProtocol {
        &self.bridge_protocol
    }

    pub fn get_chain_config(&self) -> &ChainConfig {
        &self.chain_config
    }

    pub fn get_wallet_config(&self) -> &WalletConfig {
        &self.wallet_config
    }

    pub fn get_consensus_algorithm(&self) -> &ConsensusAlgorithm {
        &self.consensus_algorithm
    }
}

pub struct CrossChainMessage {
    message_id: String,
    sender_chain_id: String,
    sender_wallet_id: String,
    recipient_chain_id: String,
    recipient_wallet_id: String,
    message_data: Vec<u8>,
}

impl CrossChainMessage {
    pub fn new(
        message_id: String,
        sender_chain_id: String,
        sender_wallet_id: String,
        recipient_chain_id: String,
        recipient_wallet_id: String,
        message_data: Vec<u8>,
    ) -> Self {
        CrossChainMessage {
            message_id,
            sender_chain_id,
            sender_wallet_id,
            recipient_chain_id,
            recipient_wallet_id,
            message_data,
        }
    }

    pub fn get_message_id(&self) -> &str {
        &self.message_id
    }

    pub fn get_sender_chain_id(&self) -> &str {
        &self.sender_chain_id
    }

    pub fn get_sender_wallet_id(&self) -> &str {
        &self.sender_wallet_id
    }

    pub fn get_recipient_chain_id(&self) -> &str {
        &self.recipient_chain_id
    }

    pub fn get_recipient_wallet_id(&self) -> &str {
        &self.recipient_wallet_id
    }

    pub fn get_message_data(&self) -> &[u8] {
        &self.message_data
    }
}
