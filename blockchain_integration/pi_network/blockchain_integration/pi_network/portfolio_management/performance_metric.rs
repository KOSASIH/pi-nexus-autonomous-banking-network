// performance_metric.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::portfolio::{Portfolio, PortfolioConfig};
use crate::asset::{Asset, AssetType};

pub trait PerformanceMetric {
    fn calculate_performance(&self, portfolio: &Portfolio) -> f64;
}

pub struct ReturnMetric {
    pubtime_period: u64,
}

impl PerformanceMetric for ReturnMetric {
    fn calculate_performance(&self, portfolio: &Portfolio) -> f64 {
        // implement return calculation logic here
        0.0
    }
}

pub struct SharpeRatio {
    pub risk_free_rate: f64,
    pub time_period: u64,
}

impl PerformanceMetric for SharpeRatio {
    fn calculate_performance(&self, portfolio: &Portfolio) -> f64 {
        // implement Sharpe Ratio calculation logic here
        0.0
    }
}
