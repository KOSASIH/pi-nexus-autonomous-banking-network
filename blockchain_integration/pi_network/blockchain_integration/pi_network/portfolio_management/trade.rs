// trade.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::asset::{Asset, AssetType};
use crate::order::{Order, OrderType};

pub struct Trade {
    pub trade_id: String,
    pub order: Arc<Order>,
    pub execution_price: f64,
    pub execution_time: u64,
}

impl Trade {
    pub fn new(trade_id: String, order: Arc<Order>, execution_price: f64, execution_time: u64) -> Self {
        Trade {
            trade_id,
            order,
            execution_price,
            execution_time,
        }
    }

    pub fn get_trade_id(&self) -> &str {
        &self.trade_id
    }

    pub fn get_order(&self) -> &Arc<Order> {
        &self.order
    }

    pub fn get_execution_price(&self) -> f64 {
        self.execution_price
    }

    pub fn get_execution_time(&self) -> u64 {
        self.execution_time
    }

    pub fn get_asset(&self) -> Asset {
        self.order.get_asset()
    }

    pub fn get_order_type(&self) -> OrderType {
        self.order.get_order_type()
    }

    pub fn get_quantity(&self) -> u64 {
        self.order.get_quantity()
    }
}
