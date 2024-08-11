// cli.rs (new)

use crate::blockchain::Blockchain;
use crate::transaction::Transaction;
use clap::{App, Arg};

fn main() {
    let app = App::new("Blockchain CLI")
        .version("1.0")
        .author("Your Name")
        .about("A command-line interface for the blockchain")
        .arg(
            Arg::with_name("command")
                .required(true)
                .index(1)
                .help("The command to execute (e.g. 'create_transaction', 'add_block', 'get_chain')"),
        );

    let matches = app.get_matches();

    let blockchain = Blockchain::new();

    match matches.value_of("command").unwrap() {
        "create_transaction" => {
            // TO DO: implement create transaction logic
            println!("Create transaction command");
        }
        "add_block" => {
            // TO DO: implement add block logic
            println!("Add block command");
        }
        "get_chain" => {
            println!("Blockchain chain:");
            for block in blockchain.chain {
                println!("{:?}", block);
            }
        }
        _ => println!("Invalid command"),
    }
}
