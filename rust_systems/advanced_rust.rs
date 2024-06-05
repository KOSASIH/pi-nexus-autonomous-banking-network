use std::sync::{Arc, Mutex};
use std::thread;

// Define a struct to hold some data
struct Data {
    value: i32,
}

// Implement Drop trait to demonstrate ownership and borrowing
impl Drop for Data {
    fn drop(&mut self) {
        println!("Dropping Data with value {}", self.value);
    }
}

fn main() {
    // Create an Arc to share ownership of Data
    let data = Arc::new(Data { value: 10 });
    let data_clone = Arc::clone(&data);

    // Spawn a new thread to demonstrate borrowing
    thread::spawn(move || {
        let mutex = Mutex::new(data_clone);
        let mut data_locked = mutex.lock().unwrap();
        data_locked.value = 20;
    });

    // Wait for the thread to finish
    thread::sleep(std::time::Duration::from_millis(100));

    // Demonstrate ownership transfer
    let data_taken = Arc::try_unwrap(data).unwrap();
    println!("Data value: {}", data_taken.value);
}
