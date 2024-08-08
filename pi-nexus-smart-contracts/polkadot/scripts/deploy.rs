use polkadot_api::{Api, ApiError};
use polkadot_primitives::{AccountId, Balance};
use polkadot_runtime::{Block, Header};
use sp_core::{crypto::Ss58Codec, H256};
use sp_runtime::{traits::Hash, MultiAddress};

// Set the Polkadot API endpoint
const API_ENDPOINT: &str = "ws://localhost:9944";

// Set the contract code and metadata
const ESCROW_CONTRACT_CODE: &[u8] = include_bytes!("../escrow.wasm");
const LENDING_CONTRACT_CODE: &[u8] = include_bytes!("../lending.wasm");
const TOKENIZED_ASSETS_CONTRACT_CODE: &[u8] = include_bytes!("../tokenized_assets.wasm");

// Set the contract metadata
const ESCROW_CONTRACT_METADATA: &str = include_str!("../escrow.metadata.json");
const LENDING_CONTRACT_METADATA: &str = include_str!("../lending.metadata.json");
const TOKENIZED_ASSETS_CONTRACT_METADATA: &str = include_str!("../tokenized_assets.metadata.json");

#[tokio::main]
async fn main() -> Result<(), ApiError> {
    // Create a new Polkadot API instance
    let api = Api::new(API_ENDPOINT).await?;

    // Deploy the escrow contract
    let escrow_contract_id = deploy_contract(
        &api,
        ESCROW_CONTRACT_CODE,
        ESCROW_CONTRACT_METADATA,
        "Escrow Contract",
    ).await?;

    // Deploy the lending contract
    let lending_contract_id = deploy_contract(
        &api,
        LENDING_CONTRACT_CODE,
        LENDING_CONTRACT_METADATA,
        "Lending Contract",
    ).await?;

    // Deploy the tokenized assets contract
    let tokenized_assets_contract_id = deploy_contract(
        &api,
        TOKENIZED_ASSETS_CONTRACT_CODE,
        TOKENIZED_ASSETS_CONTRACT_METADATA,
        "Tokenized Assets Contract",
    ).await?;

    // Print the contract IDs
    println!("Escrow Contract ID: {}", escrow_contract_id);
    println!("Lending Contract ID: {}", lending_contract_id);
    println!("Tokenized Assets Contract ID: {}", tokenized_assets_contract_id);

    Ok(())
}

async fn deploy_contract(
    api: &Api,
    code: &[u8],
    metadata: &str,
    name: &str,
) -> Result<H256, ApiError> {
    // Create a new contract deployment proposal
    let proposal = api
        .proposal_create_contract(code, metadata, name)
        .await?;

    // Submit the proposal to the Polkadot network
    let tx_hash = api.submit_proposal(proposal).await?;

    // Wait for the proposal to be executed
    api.wait_for_tx(tx_hash).await?;

    // Get the contract ID from the proposal
    let contract_id = proposal.contract_id;

    Ok(contract_id)
}
