use stellar_sdk::{Client, Horizon, Network};
use stellar_sdk::types::{Account, Asset, Transaction};

pub struct AdvancedStellarSDK {
    client: Client,
    horizon: Horizon,
    network: Network,
}

impl AdvancedStellarSDK {
    pub fn new(horizon_url: &str, network_passphrase: &str) -> Self {
        let client = Client::new(horizon_url);
        let horizon = Horizon::new(horizon_url);
        let network = Network::new(network_passphrase);
        AdvancedStellarSDK { client, horizon, network }
    }

    pub fn create_account(&self, seed: &str, account_name: &str) -> Result<Transaction, Error> {
        let kp = KeyPair::from_seed(seed);
        let account = self.client.account(kp.address())?;
        if account.is_none() {
            let tx = Transaction::new(
                kp.address(),
                vec![Operation::CreateAccount {
                    destination: kp.address(),
                    starting_balance: "100.0",
                }],
            );
            self.client.submit_transaction(tx)?;
            Ok(tx)
        } else {
            Err(Error::AccountAlreadyExists)
        }
    }

    pub fn issue_asset(&self, asset_code: &str, asset_issuer: &str, amount: &str) -> Result<Transaction, Error> {
        let tx = Transaction::new(
            asset_issuer,
            vec![Operation::Payment {
                destination: asset_issuer,
                asset: Asset::new(asset_code, asset_issuer),
                amount: amount.to_string(),
            }],
        );
        self.client.submit_transaction(tx)?;
        Ok(tx)
    }

    pub fn create_trustline(&self, source_account: &str, asset_code: &str, asset_issuer: &str) -> Result<Transaction, Error> {
        let tx = Transaction::new(
            source_account,
            vec![Operation::ChangeTrust {
                asset: Asset::new(asset_code, asset_issuer),
            }],
        );
        self.client.submit_transaction(tx)?;
        Ok(tx)
    }

    pub fn path_payment(&self, source_account: &str, destination_account: &str, asset_code: &str, amount: &str) -> Result<Transaction, Error> {
        let tx = Transaction::new(
            source_account,
            vec![Operation::PathPayment {
                send_asset: Asset::new(asset_code, source_account),
                send_amount: amount.to_string(),
                destination: destination_account,
                dest_asset: Asset::new(asset_code, destination_account),
            }],
        );
        self.client.submit_transaction(tx)?;
        Ok(tx)
    }
}
