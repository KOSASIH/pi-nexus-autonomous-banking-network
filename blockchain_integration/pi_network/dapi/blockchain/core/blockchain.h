#ifndef BLOCKCHAIN_H
#define BLOCKCHAIN_H

#include <vector>
#include "block.h"

class Blockchain {
public:
  Blockchain();
  ~Blockchain();

  void add_block(const std::vector<std::string>& transactions);
  const std::vector<Block>& get_chain() const;

private:
  std::vector<Block> chain_;
  uint32_t difficulty_;
};

#endif  // BLOCKCHAIN_H
