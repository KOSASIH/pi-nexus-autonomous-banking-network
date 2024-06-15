// rebalancer.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};

pub struct Rebalancer {
    rebalancing_strategies: HashMap<String, Arc<RebalancingStrategy>>,
}

impl Rebalancer {
    pub fn new() -> Self {
        Rebalancer {
            rebalancing_strategies: HashMap::new(),
        }
    }

    pub fn add_rebalancing_strategy(&mut self, strategy_id: String, strategy: Arc<RebalancingStrategy>) {
        self.rebalancing_strategies.insert(strategy_id, strategy);
    }

    pub fn get_rebalancing_strategy(&self, strategy_id: &str) -> Option<Arc<RebalancingStrategy>> {
        self.rebalancing_strategies.get(strategy_id).cloned()
    }

    pub fn rebalance_portfolio(&self, portfolio: &mut Portfolio) {
        // implement rebalancing logic here
    }
}
