#ifndef CRYPTO_H
#define CRYPTO_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

// Hash function types
enum class HashType {
  SHA256,
  SHA512,
  BLAKE2b,
  ARGON2
};

// Cryptographic error types
enum class CryptoError {
  INVALID_INPUT,
  HASH_FAILURE,
  ENCRYPTION_FAILURE,
  DECRYPTION_FAILURE
};

// Hash-based cryptography class
class Crypto {
public:
  Crypto(const std::string& password, HashType hash_type = HashType::SHA256);
  ~Crypto();

  // Hash a message
  std::string hash(const std::string& message);

  // Encrypt a message
  std::string encrypt(const std::string& message);

  // Decrypt a message
  std::string decrypt(const std::string& ciphertext);

  // Generate a random salt
  std::string generate_salt();

  // Get the hash type
  HashType get_hash_type();

private:
  std::string password_;
  HashType hash_type_;
  std::string salt_;
};

// Exception class for cryptographic errors
class CryptoException : public std::exception {
public:
  CryptoException(CryptoError error);
  const char* what() const throw();

private:
  CryptoError error_;
};

#endif // CRYPTO_H
