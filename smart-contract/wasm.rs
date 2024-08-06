// wasm.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SmartContract {
    pub storage: Vec<u8>,
}

#[wasm_bindgen]
impl SmartContract {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { storage: Vec::new() }
    }

    #[wasm_bindgen(js_name = "store")]
    pub fn store(&mut self, key: &str, value: &str) {
        self.storage.push(format!("{}={}", key, value).as_bytes().to_vec());
    }

    #[wasm_bindgen(js_name = "retrieve")]
    pub fn retrieve(&self, key: &str) -> Option<String> {
        for entry in &self.storage {
            let entry_str = std::str::from_utf8(entry).unwrap();
            let parts: Vec<&str> = entry_str.split("=").collect();
            if parts[0] == key {
                return Some(parts[1].to_string());
            }
        }
        None
    }
}
