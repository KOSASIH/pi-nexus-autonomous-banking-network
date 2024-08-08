use solana_client::{
    rpc_client::RpcClient,
    transaction::Transaction,
};
use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
};
use solana_sdk::{
    signature::{Keypair, Signer},
    transaction::TransactionError,
};
use std::error::Error;

// Define the contract program ID
const PROGRAM_ID: &str = "TokenizedAssets1111111111111111111111111111111111";

// Define the deploy function
fn deploy(
    rpc_url: &str,
    keypair_path: &str,
    program_path: &str,
) -> Result<(), Box<dyn Error>> {
    // Create a new RPC client
    let rpc_client = RpcClient::new(rpc_url);

    // Load the keypair from the file
    let keypair = Keypair::from_file(keypair_path)?;

    // Load the program from the file
    let program_data = std::fs::read(program_path)?;
    let program_id = Pubkey::new_unique();

    // Create a new account for the program
    let account = rpc_client.get_account(&program_id)?;
    if account.is_none() {
        msg!("Creating new program account");
        let lamports = rpc_client.get_minimum_balance_for_rent_exemption(program_data.len())?;
        let tx = Transaction::new_signed_with_payer(
            &[rpc_client.request_airdrop(&keypair.pubkey(), lamports)?],
            Some(&keypair.pubkey()),
            &[&keypair],
            rpc_client.get_latest_blockhash()?,
        );
        rpc_client.send_and_confirm_transaction(&tx)?;
    }

    // Upload the program to the account
    msg!("Uploading program");
    let tx = Transaction::new_signed_with_payer(
        &[rpc_client.create_account(
            &keypair.pubkey(),
            &program_id,
            program_data.len() as u64,
            lamports,
        )?],
        Some(&keypair.pubkey()),
        &[&keypair],
        rpc_client.get_latest_blockhash()?,
    );
    rpc_client.send_and_confirm_transaction(&tx)?;

    // Set the program ID
    msg!("Setting program ID");
    let tx = Transaction::new_signed_with_payer(
        &[rpc_client.write_account(
            &keypair.pubkey(),
            &program_id,
            &program_data,
        )?],
        Some(&keypair.pubkey()),
        &[&keypair],
        rpc_client.get_latest_blockhash()?,
    );
    rpc_client.send_and_confirm_transaction(&tx)?;

    Ok(())
}

fn main() {
    if let Err(err) = deploy(
        "https://api.devnet.solana.com",
        "path/to/keypair.json",
        "path/to/program.so",
    ) {
        msg!("Error: {}", err);
        std::process::exit(1);
    }
}
