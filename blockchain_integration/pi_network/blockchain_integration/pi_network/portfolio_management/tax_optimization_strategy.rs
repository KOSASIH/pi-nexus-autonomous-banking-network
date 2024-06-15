// tax_optimization_strategy.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};

pub trait TaxOptimizationStrategy {
    fn optimize_tax(&self, portfolio: &mut Portfolio);
}

pub struct LongTermCapitalGains {
    pub tax_rate: f64,
}

impl TaxOptimizationStrategy for LongTermCapitalGains {
    fn optimize_tax(&self, portfolio: &mut Portfolio) {
        // implement long-term capital gains tax optimization logic here
    }
}

pub struct ShortTermCapitalGains {
    pub tax_rate: f64,
}

impl TaxOptimizationStrategy for ShortTermCapitalGains {
    fn optimize_tax(&self, portfolio: &mut Portfolio) {
        // implement short-term capital gains tax optimization logic here
    }
}
