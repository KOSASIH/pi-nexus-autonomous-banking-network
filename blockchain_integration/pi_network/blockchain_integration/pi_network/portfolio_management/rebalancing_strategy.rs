// rebalancing_strategy.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};

pub trait RebalancingStrategy {
    fn rebalance_portfolio(&self, portfolio: &mut Portfolio);
}

pub struct PeriodicRebalancing {
    pub rebalancing_frequency: u64,
}

impl RebalancingStrategy for PeriodicRebalancing {
    fn rebalance_portfolio(&self, portfolio: &mut Portfolio) {
        // implement periodic rebalancing logic here
    }
}

pub struct ThresholdRebalancing {
    pub threshold: f64,
}

impl RebalancingStrategy for ThresholdRebalancing {
    fn rebalance_portfolio(&self, portfolio: &mut Portfolio) {
        // implement threshold rebalancing logic here
    }
}
