// utils/errors.rs: Error handling utility implementation

use std::fmt;

#[derive(Debug)]
pub enum Error {
    InternalError(String),
    InvalidInput(String),
    UnexpectedError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InternalError(message) => write!(f, "Internal error: {}", message),
            Error::InvalidInput(message) => write!(f, "Invalid input: {}", message),
            Error::UnexpectedError(message) => write!(f, "Unexpected error: {}", message),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn internal_error<T>(message: T) -> Result<T> {
    Err(Error::InternalError(message.to_string()))
}

pub fn invalid_input<T>(message: T) -> Result<T> {
    Err(Error::InvalidInput(message.to_string()))
}

pub fn unexpected_error<T>(message: T) -> Result<T> {
    Err(Error::UnexpectedError(message.to_string()))
}
