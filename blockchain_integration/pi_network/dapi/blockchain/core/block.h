#ifndef BLOCK_H
#define BLOCK_H

#include <string>
#include <vector>
#include <cstdint>
#include <openssl/sha.h>

class Block {
public:
  Block(uint32_t index, const std::string& previous_hash, const std::vector<std::string>& transactions);
  ~Block();

  std::string get_hash() const;
  uint32_t get_index() const;
  std::string get_previous_hash() const;
  const std::vector<std::string>& get_transactions() const;

  void set_nonce(uint32_t nonce);
  uint32_t get_nonce() const;

  void mine_block(uint32_t difficulty);

private:
  uint32_t index_;
  std::string previous_hash_;
  std::vector<std::string> transactions_;
  uint32_t nonce_;
  std::string hash_;
};

#endif  // BLOCK_H
