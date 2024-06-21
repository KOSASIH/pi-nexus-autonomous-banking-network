// Node implementation with advanced networking and communication features
use crate::blockchain::{Block, Blockchain};
use crate::network::{Network, NodeId};
use tokio::{net::TcpListener, prelude::*};

pub struct Node {
    id: NodeId,
    blockchain: Blockchain,
    network: Network,
    listener: TcpListener,
}

impl Node {
    pub async fn new(id: NodeId, blockchain: Blockchain, network: Network) -> Result<Self, String> {
        let listener = TcpListener::bind("0.0.0.0:8080").await?;
        Ok(Node {
            id,
            blockchain,
            network,
            listener,
        })
    }

    pub async fn start(&mut self) -> Result<(), String> {
        // Start the node and begin listening for incoming connections
        self.listener.listen(10).await?;
        Ok(())
    }

    pub async fn broadcast_block_proposal(&mut self, block: Block) -> Result<(), String> {
        // Broadcast the block proposal to the network
        for node in self.network.nodes() {
            if node.id != self.id {
                node.send_block_proposal(block.clone()).await?;
            }
        }
        Ok(())
    }
}
