// position.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::asset::{Asset, AssetType};

pub struct Position {
    position_id: String,
    asset: Asset,
    quantity: u64,
}

impl Position {
    pub fn new(position_id: String, asset: Asset, quantity: u64) -> Self {
        Position {
            position_id,
            asset,
            quantity,
        }
    }

    pub fn get_position_id(&self) -> &str {
        &self.position_id
    }

    pub fn get_asset(&self) -> &Asset {
        &self.asset
    }

    pub fn get_quantity(&self) -> u64 {
        self.quantity
    }
}
