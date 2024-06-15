// order.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::asset::{Asset, AssetType};

pub enum OrderType {
    Buy,
    Sell,
}

pub struct Order {
    order_id: String,
    asset: Asset,
    quantity: u64,
    order_type: OrderType,
}

impl Order {
    pub fn new(order_id: String, asset: Asset, quantity: u64, order_type: OrderType) -> Self {
        Order {
            order_id,
            asset,
            quantity,
            order_type,
        }
    }

    pub fn get_order_id(&self) -> &str {
        &self.order_id
    }

    pub fn get_asset(&self) -> &Asset {
        &self.asset
    }

    pub fn get_quantity(&self) -> u64 {
        self.quantity
    }

    pub fn get_order_type(&self) -> &OrderType {
        &self.order_type
    }
}
