use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

struct Cybersecurity {
    rules: HashMap<String, String>,
}

impl Cybersecurity {
    fn new() -> Self {
        let mut rules = HashMap::new();
        // Load rules from a file
        let file = File::open("rules.txt").unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            let mut parts = line.split(":");
            let key = parts.next().unwrap();
            let value = parts.next().unwrap();
            rules.insert(key.to_string(), value.to_string());
        }
        Cybersecurity { rules }
    }

    fn detect_threats(&self, data: &str) -> Vec<String> {
        let mut threats = Vec::new();
        for (key, value) in &self.rules {
            if data.contains(key) {
                threats.push(value.clone());
            }
        }
        threats
    }
}

// Example usage:
let cybersecurity = Cybersecurity::new();
let data = "This is a sample data with a threat";
let threats = cybersecurity.detect_threats(data);
for threat in threats {
    println!("Detected threat: {}", threat);
}
