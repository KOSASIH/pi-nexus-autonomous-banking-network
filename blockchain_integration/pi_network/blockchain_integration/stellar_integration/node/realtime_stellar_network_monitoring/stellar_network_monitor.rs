use grpc::{Client, Request, Response};
use stellar_sdk::types::{Transaction, Block};

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
        let request = Request::new("listen", "transactions");
        let response = self.client.unary(request).await?;
        let transactions: Vec<Transaction> = response.get_message()?;
        for tx in transactions {
            println!("Received transaction: {}", tx.hash);
        }
    }
}
