use tokio::prelude::*;
use tokio::stream::StreamExt;
use websocket::{Client, Message};

struct StellarNetworkMonitor {
    client: Client,
    horizon_url: String,
}

impl StellarNetworkMonitor {
    async fn new(horizon_url: &str) -> Self {
        let client = Client::new(horizon_url).await?;
        StellarNetworkMonitor { client, horizon_url: horizon_url.to_string() }
    }

   async fn start_monitoring(&mut self) {
        let (mut ws_stream, _) = self.client.connect("/websocket").await?;
        ws_stream.send(Message::text(r#"{"command": "listen", "stream": "transactions"}"#)).await?;
        while let Some(message) = ws_stream.next().await {
            match message {
                Message::Text(text) => {
                    let tx: Transaction = serde_json::from_str(&text).unwrap();
                    println!("Received transaction: {}", tx.hash);
                }
                _ => {}
            }
        }
    }
}
