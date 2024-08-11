// network.rs (new)

use crate::blockchain::Blockchain;
use tokio::net::TcpListener;
use tokio::prelude::*;

pub struct Node {
    blockchain: Blockchain,
    listener: TcpListener,
}

impl Node {
    pub async fn new(blockchain: Blockchain, addr: &str) -> Self {
        let listener = TcpListener::bind(addr).await.unwrap();
        Node { blockchain, listener }
    }

    pub async fn start(self) {
        println!("Node started on {}", self.listener.local_addr().unwrap());
        while let Ok((mut socket, _)) = self.listener.accept().await {
            tokio::spawn(async move {
                // TO DO: implement handle incoming connection logic
                println!("Incoming connection");
            });
        }
    }
}
