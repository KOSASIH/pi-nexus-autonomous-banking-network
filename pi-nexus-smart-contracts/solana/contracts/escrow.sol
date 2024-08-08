use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
};
use spl_token::{error::TokenError, state::Account};

// Define the Escrow contract
pub struct Escrow {
    // The account that owns the escrow
    owner: Pubkey,
    // The account that holds the escrowed tokens
    escrow_account: Pubkey,
    // The amount of tokens escrowed
    amount: u64,
}

// Define the Escrow contract's instructions
enum EscrowInstruction {
    // Initialize the escrow contract
    InitEscrow,
    // Deposit tokens into the escrow
    DepositTokens,
    // Withdraw tokens from the escrow
    WithdrawTokens,
}

// Define the Escrow contract's entry point
entrypoint!(process_instruction);

fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    // Unpack the instruction data
    let instruction = EscrowInstruction::unpack(instruction_data)?;

    // Handle the instruction
    match instruction {
        EscrowInstruction::InitEscrow => {
            init_escrow(program_id, accounts)?;
        }
        EscrowInstruction::DepositTokens => {
            deposit_tokens(program_id, accounts)?;
        }
        EscrowInstruction::WithdrawTokens => {
            withdraw_tokens(program_id, accounts)?;
        }
    }

    Ok(())
}

// Initialize the escrow contract
fn init_escrow(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let escrow_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;

    // Create a new escrow account
    let escrow_account = Account::unpack(&escrow_account_info.data.borrow())?;
    if escrow_account.owner != *program_id {
        msg!("Error: Escrow account already initialized");
        return Err(ProgramError::Custom(1));
    }

    // Initialize the escrow account
    escrow_account.owner = *program_id;
    escrow_account.amount = 0;
    escrow_account.data.borrow_mut()[..].fill(0);

    // Create a new token account
    let token_account = Account::unpack(&token_program_account_info.data.borrow())?;
    if token_account.owner != *program_id {
        msg!("Error: Token account already initialized");
        return Err(ProgramError::Custom(2));
    }

    // Initialize the token account
    token_account.owner = *program_id;
    token_account.amount = 0;
    token_account.data.borrow_mut()[..].fill(0);

    Ok(())
}

// Deposit tokens into the escrow
fn deposit_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let escrow_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

    // Unpack the accounts
    let escrow_account = Account::unpack(&escrow_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

    // Check that the owner is correct
    if owner_account_info.key != &escrow_account.owner {
        msg!("Error: Owner mismatch");
        return Err(ProgramError::Custom(3));
    }

    // Check that the token account is correct
    if token_account_info.key != &token_account.owner {
        msg!("Error: Token account mismatch");
        return Err(ProgramError::Custom(4));
    }

    // Deposit tokens into the escrow
    escrow_account.amount += token_account.amount;
    token_account.amount = 0;

    Ok(())
}

// Withdraw tokens from the escrow
fn withdraw_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let escrow_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

    // Unpack the accounts
    let escrow_account = Account::unpack(&escrow_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

    // Check that the owner is correct
    if owner_account_info.key != &escrow_account.owner {
        msg!("Error: Owner mismatch");
        return Err(ProgramError::Custom(5));
    }

    // Check that the token account is correct
    if token_account_info.key != &token_account.owner {
        msg!("Error: Token account mismatch");
        return Err(ProgramError::Custom(6));
    }

    // Check that the escrow has sufficient tokens
    if escrow_account.amount < token_account.amount {
        msg!("Error: Insufficient tokens in escrow");
        return Err(ProgramError::Custom(7));
    }

    // Withdraw tokens from the escrow
    escrow_account.amount -= token_account.amount;
    token_account.amount += token_account.amount;

    Ok(())
}

// Implement the EscrowInstruction enum
impl EscrowInstruction {
    fn unpack(instruction_data: &[u8]) -> Result<EscrowInstruction, ProgramError> {
        if instruction_data.len() != 1 {
            msg!("Error: Invalid instruction data");
            return Err(ProgramError::InvalidInstructionData);
        }

        match instruction_data[0] {
            0 => Ok(EscrowInstruction::InitEscrow),
            1 => Ok(EscrowInstruction::DepositTokens),
            2 => Ok(EscrowInstruction::WithdrawTokens),
            _ => {
                msg!("Error: Unknown instruction");
                Err(ProgramError::InvalidInstruction)
            }
        }
    }
}
