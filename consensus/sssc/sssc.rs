use std::collections::HashMap;

struct SSSCA {
    shards: HashMap<String, Vec<String>>,
}

impl SSSCA {
    fn new() -> Self {
        SSSCA {
            shards: HashMap::new(),
        }
    }

    fn add_shard(&mut self, shard_id: String, nodes: Vec<String>) {
        self.shards.insert(shard_id, nodes);
    }

    fn get_shard(&self, shard_id: String) -> Option<&Vec<String>> {
        self.shards.get(&shard_id)
    }
}
