// network.rs (update)

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
                let mut buf = [0; 1024];
                let n = socket.read(&mut buf).await.unwrap();
                let message = String::from_utf8_lossy(&buf[..n]);
                match message.as_str() {
                    "get_chain" => {
                        let chain = self.blockchain.chain.clone();
                        let response = serde_json::to_string(&chain).unwrap();
                        socket.write(response.as_bytes()).await.unwrap();
                    }
                    _ => println!("Invalid message"),
                }
            });
        }
    }
}
