#include "block.h"
#include <openssl/sha.h>
#include <iostream>

Block::Block(uint32_t index, const std::string& previous_hash, const std::vector<std::string>& transactions)
    : index_(index), previous_hash_(previous_hash), transactions_(transactions), nonce_(0), hash_("") {}

Block::~Block() {}

std::string Block::get_hash() const {
  return hash_;
}

uint32_t Block::get_index() const {
  return index_;
}

std::string Block::get_previous_hash() const {
  return previous_hash_;
}

const std::vector<std::string>& Block::get_transactions() const {
  return transactions_;
}

void Block::set_nonce(uint32_t nonce) {
  nonce_ = nonce;
}

uint32_t Block::get_nonce() const {
  return nonce_;
}

void Block::mine_block(uint32_t difficulty) {
  char buffer[SHA256_DIGEST_LENGTH * 2 + 1];
  uint32_t nonce = 0;
  while (true) {
    std::string header = std::to_string(index_) + previous_hash_ + std::to_string(nonce) + std::to_string(difficulty);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256_ctx;
    SHA256_Init(&sha256_ctx);
    SHA256_Update(&sha256_ctx, header.c_str(), header.size());
    SHA256_Final(hash, &sha256_ctx);
    sprintf(buffer, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
            hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]);
    std::string hash_str(buffer);
    if (hash_str.substr(0, difficulty) == std::string(difficulty, '0')) {
      hash_ = hash_str;
      nonce_ = nonce;
      break;
    }
    nonce++;
  }
}
