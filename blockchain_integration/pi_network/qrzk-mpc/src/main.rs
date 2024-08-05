// main.rs: Main entry point for the QRZK-MPC system

use std::env;
use std::fs;
use std::path::Path;

mod lib;

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let config_file = args.get(1).expect("Expected config file path as argument");
    let input_file = args.get(2).expect("Expected input file path as argument");
    let output_file = args.get(3).expect("Expected output file path as argument");

    // Load configuration from file
    let config = lib::Config::from_file(config_file).expect("Failed to load config from file");

    // Load input data from file
    let input_data = fs::read_to_string(input_file).expect("Failed to read input file");
    let input: Vec<u8> = input_data.into_bytes();

    // Create a new QRZK-MPC instance
    let mut qrzk_mpc = lib::QRZKMPC::new(config);

    // Generate a proof for the input data
    let proof = qrzk_mpc.generate_proof(input).expect("Failed to generate proof");

    // Save the proof to file
    fs::write(output_file, proof.to_bytes()).expect("Failed to write proof to file");

    println!("Proof generated and saved to {}", output_file);
}
