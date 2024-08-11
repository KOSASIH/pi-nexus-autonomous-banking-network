// cli.rs (update)

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

    let mut blockchain = Blockchain::new();

    match matches.value_of("command").unwrap() {
        "create_transaction" => {
            let sender = "Alice";
            let recipient = "Bob";
            let amount = 10;
            let transaction = Transaction::new(sender, recipient, amount);
            blockchain.add_transaction(transaction);
            println!("Transaction created and added to the blockchain");
        }
        "add_block" => {
            blockchain.add_block(blockchain.get_transactions());
            println!("Block added to the blockchain");
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
