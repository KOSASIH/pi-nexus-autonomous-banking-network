#include "crypto.h"
#include <openssl/sha.h>
#include <openssl/err.h>
#include <blake2.h>
#include <argon2.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

// Hash function implementations
std::string sha256(const std::string& message) {
  unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha256_ctx;
  SHA256_Init(&sha256_ctx);
  SHA256_Update(&sha256_ctx, message.c_str(), message.size());
  SHA256_Final(sha256_hash, &sha256_ctx);
  return std::string((char*)sha256_hash, SHA256_DIGEST_LENGTH);
}

std::string sha512(const std::string& message) {
  unsigned char sha512_hash[SHA512_DIGEST_LENGTH];
  SHA512_CTX sha512_ctx;
  SHA512_Init(&sha512_ctx);
  SHA512_Update(&sha512_ctx, message.c_str(), message.size());
  SHA512_Final(sha512_hash, &sha512_ctx);
  return std::string((char*)sha512_hash, SHA512_DIGEST_LENGTH);
}

std::string blake2b(const std::string& message) {
  unsigned char blake2b_hash[BLAKE2B_OUTBYTES];
  blake2b_state blake2b_ctx;
  blake2b_init(&blake2b_ctx, BLAKE2B_OUTBYTES);
  blake2b_update(&blake2b_ctx, message.c_str(), message.size());
  blake2b_final(&blake2b_ctx, blake2b_hash, BLAKE2B_OUTBYTES);
  return std::string((char*)blake2b_hash, BLAKE2B_OUTBYTES);
}

std::string argon2(const std::string& message) {
  unsigned char argon2_hash[ARGON2_OUTBYTES];
  argon2_context argon2_ctx;
  argon2_init(&argon2_ctx, ARGON2_OUTBYTES);
  argon2_update(&argon2_ctx, message.c_str(), message.size());
  argon2_final(&argon2_ctx, argon2_hash, ARGON2_OUTBYTES);
  return std::string((char*)argon2_hash, ARGON2_OUTBYTES);
}

// Crypto class implementation
Crypto::Crypto(const std::string& password, HashType hash_type) : password_(password), hash_type_(hash_type) {
  // Generate a random salt
  salt_ = generate_salt();
}

Crypto::~Crypto() {}

std::string Crypto::hash(const std::string& message) {
  switch (hash_type_) {
    case HashType::SHA256:
      return sha256(message + salt_);
    case HashType::SHA512:
      return sha512(message + salt_);
    case HashType::BLAKE2b:
      return blake2b(message + salt_);
    case HashType::ARGON2:
      return argon2(message + salt_);
    default:
      throw CryptoException(CryptoError::INVALID_INPUT);
  }
}

std::string Crypto::encrypt(const std::string& message) {
  // Encrypt the message using a hash-based encryption scheme
  std::string ciphertext;
  // TO DO: implement encryption algorithm
  return ciphertext;
}

std::string Crypto::decrypt(const std::string& ciphertext) {
  // Decrypt the ciphertext using a hash-based decryption scheme
  std::string plaintext;
  // TO DO: implement decryption algorithm
  return plaintext;
}

std::string Crypto::generate_salt() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist('a', 'z');
  std::string salt;
  for (int i = 0; i < 16; i++) {
    salt += dist(mt);
  }
  return salt;
}

HashType Crypto::get_hash_type() {
  return hash_type_;
}

// CryptoException implementation
CryptoException::CryptoException(CryptoError error) : error_(error) {}

const char* CryptoException::what() const throw() {
  switch (error_) {
    case CryptoError::INVALID_INPUT:
      return "Invalid input";
    case CryptoError::HASH_FAILURE:
      return "Hash failure";
    case CryptoError::ENCRYPTION_FAILURE:
      return "Encryption failure";
    case CryptoError::DECRYPTION_FAILURE:
      return "Decryption failure";
    default:
      return "Unknown error";
  }
}
