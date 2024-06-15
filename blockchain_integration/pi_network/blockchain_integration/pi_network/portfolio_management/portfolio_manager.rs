// portfolio_manager.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};
use crate::wallet::{Wallet, WalletConfig};

pub struct PortfolioManager {
    portfolios: HashMap<String, Arc<Portfolio>>,
    wallet_manager: Arc<WalletManager>,
}

impl PortfolioManager {
    pub fn new(wallet_manager: Arc<WalletManager>) -> Self {
        PortfolioManager {
            portfolios: HashMap::new(),
            wallet_manager,
        }
    }

    pub fn create_portfolio(&mut self, portfolio_config: PortfolioConfig) -> Arc<Portfolio> {
        let portfolio = Portfolio::new(portfolio_config);
        self.portfolios.insert(portfolio.get_portfolio_id().to_string(), Arc::new(portfolio));
        Arc::clone(&self.portfolios[&portfolio.get_portfolio_id().to_string()])
    }

    pub fn get_portfolio(&self, portfolio_id: &str) -> Option<Arc<Portfolio>> {
        self.portfolios.get(portfolio_id).cloned()
    }

    pub fn get_all_portfolios(&self) -> Vec<Arc<Portfolio>> {
        self.portfolios.values().cloned().collect()
    }
}
