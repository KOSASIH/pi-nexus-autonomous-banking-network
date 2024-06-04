use zokrates_core::{compile, setup};

// Define the zk-SNARK circuitlet circuit = /* define the circuit */;

// Compile the circuit
let compiled_circuit = compile(circuit).unwrap();

// Generate a proof
let proof = setup(compiled_circuit).unwrap();
