use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
};
use spl_token::{error::TokenError, state::Account};

// Define the Lending contract
pub struct Lending {
    // The account that owns the lending contract
    owner: Pubkey,
    // The account that holds the lent tokens
    lent_account: Pubkey,
    // The amount of tokens lent
    amount: u64,
    // The interest rate
    interest_rate: u64,
}

// Define the Lending contract's instructions
enum LendingInstruction {
    // Initialize the lending contract
    InitLending,
    // Deposit tokens into the lending contract
    DepositTokens,
    // Withdraw tokens from the lending contract
    WithdrawTokens,
    // Borrow tokens from the lending contract
    BorrowTokens,
    // Repay tokens to the lending contract
    RepayTokens,
}

// Define the Lending contract's entry point
entrypoint!(process_instruction);

fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    // Unpack the instruction data
    let instruction = LendingInstruction::unpack(instruction_data)?;

    // Handle the instruction
    match instruction {
        LendingInstruction::InitLending => {
            init_lending(program_id, accounts)?;
        }
        LendingInstruction::DepositTokens => {
            deposit_tokens(program_id, accounts)?;
        }
        LendingInstruction::WithdrawTokens => {
            withdraw_tokens(program_id, accounts)?;
        }
        LendingInstruction::BorrowTokens => {
            borrow_tokens(program_id, accounts)?;
        }
        LendingInstruction::RepayTokens => {
            repay_tokens(program_id, accounts)?;
        }
    }

    Ok(())
}

// Initialize the lending contract
fn init_lending(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let lent_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;

    // Create a new lent account
    let lent_account = Account::unpack(&lent_account_info.data.borrow())?;
    if lent_account.owner != *program_id {
        msg!("Error: Lent account already initialized");
        return Err(ProgramError::Custom(1));
    }

    // Initialize the lent account
    lent_account.owner = *program_id;
    lent_account.amount = 0;
    lent_account.data.borrow_mut()[..].fill(0);

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

// Deposit tokens into the lending contract
fn deposit_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let lent_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

 // Unpack the accounts
    let lent_account = Account::unpack(&lent_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

 // Check that the owner is correct
    if owner_account_info.key != &lent_account.owner {
        msg!("Error: Owner mismatch");
        return Err(ProgramError::Custom(3));
    }

  // Check that the token account is correct
    if token_account_info.key != &token_account.owner {
        msg!("Error: Token account mismatch");
        return Err(ProgramError::Custom(4));
    }

   // Deposit tokens into the lending contract
    lent_account.amount += token_account.amount;
    token_account.amount = 0;

    Ok(())
}

// Withdraw tokens from the lending contract
fn withdraw_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let owner_account_info = next_account_info(accounts)?;
    let lent_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

    // Unpack the accounts
    let lent_account = Account::unpack(&lent_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

    // Check that the owner is correct
    if owner_account_info.key != &lent_account.owner {
        msg!("Error: Owner mismatch");
        return Err(ProgramError::Custom(5));
    }

    // Check that the token account is correct
    if token_account_info.key != &token_account.owner {
        msg!("Error: Token account mismatch");
        return Err(ProgramError::Custom(6));
    }

    // Check that the lending contract has sufficient tokens
    if lent_account.amount < token_account.amount {
        msg!("Error: Insufficient tokens in lending contract");
        return Err(ProgramError::Custom(7));
    }

    // Withdraw tokens from the lending contract
    lent_account.amount -= token_account.amount;
    token_account.amount += token_account.amount;

    Ok(())
}

// Borrow tokens from the lending contract
fn borrow_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let borrower_account_info = next_account_info(accounts)?;
    let lent_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

    // Unpack the accounts
    let lent_account = Account::unpack(&lent_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

    // Check that the borrower is correct
    if borrower_account_info.key != &token_account.owner {
        msg!("Error: Borrower mismatch");
        return Err(ProgramError::Custom(8));
    }

    // Check that the lending contract has sufficient tokens
    if lent_account.amount < token_account.amount {
        msg!("Error: Insufficient tokens in lending contract");
        return Err(ProgramError::Custom(9));
    }

    // Borrow tokens from the lending contract
    lent_account.amount -= token_account.amount;
    token_account.amount += token_account.amount;

    Ok(())
}

// Repay tokens to the lending contract
fn repay_tokens(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // Get the account infos
    let borrower_account_info = next_account_info(accounts)?;
    let lent_account_info = next_account_info(accounts)?;
    let token_program_account_info = next_account_info(accounts)?;
    let token_account_info = next_account_info(accounts)?;

    // Unpack the accounts
    let lent_account = Account::unpack(&lent_account_info.data.borrow())?;
    let token_account = Account::unpack(&token_account_info.data.borrow())?;

    // Check that the borrower is correct
    if borrower_account_info.key != &token_account.owner {
        msg!("Error: Borrower mismatch");
        return Err(ProgramError::Custom(10));
    }

    // Check that the token account has sufficient tokens
    if token_account.amount < lent_account.amount {
        msg!("Error: Insufficient tokens in token account");
        return Err(ProgramError::Custom(11));
    }

    // Repay tokens to the lending contract
    lent_account.amount += token_account.amount;
    token_account.amount -= token_account.amount;

    Ok(())
}

// Implement the LendingInstruction enum
impl LendingInstruction {
    fn unpack(instruction_data: &[u8]) -> Result<LendingInstruction, ProgramError> {
        if instruction_data.len() != 1 {
            msg!("Error: Invalid instruction data");
            return Err(ProgramError::InvalidInstructionData);
        }

        match instruction_data[0] {
            0 => Ok(LendingInstruction::InitLending),
            1 => Ok(LendingInstruction::DepositTokens),
            2 => Ok(LendingInstruction::WithdrawTokens),
            3 => Ok(LendingInstruction::BorrowTokens),
            4 => Ok(LendingInstruction::RepayTokens),
            _ => {
                msg!("Error: Unknown instruction");
                Err(ProgramError::InvalidInstruction)
            }
        }
    }
}
