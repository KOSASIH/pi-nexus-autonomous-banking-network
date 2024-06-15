// tax_optimizer.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};

pub struct TaxOptimizer {
   tax_optimization_strategies: HashMap<String, Arc<TaxOptimizationStrategy>>,
}

impl TaxOptimizer {
    pub fn new() -> Self {
        TaxOptimizer {
            tax_optimization_strategies: HashMap::new(),
        }
    }

    pub fn add_tax_optimization_strategy(&mut self, strategy_id: String, strategy: Arc<TaxOptimizationStrategy>) {
        self.tax_optimization_strategies.insert(strategy_id, strategy);
    }

    pub fn get_tax_optimization_strategy(&self, strategy_id: &str) -> Option<Arc<TaxOptimizationStrategy>> {
        self.tax_optimization_strategies.get(strategy_id).cloned()
    }

    pub fn optimize_tax(&self, portfolio: &mut Portfolio) {
        // implement tax optimization logic here
    }
}
