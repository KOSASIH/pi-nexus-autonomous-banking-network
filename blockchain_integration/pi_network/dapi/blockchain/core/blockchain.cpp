#include "blockchain.h"
#include <iostream>

Blockchain::Blockchain() : difficulty_(4) {
  chain_.emplace_back(0, "0", {});
}

Blockchain::~Blockchain() {}

void Blockchain::add_block(const std::vector<std::string>& transactions) {
  Block new_block(chain_.size(), chain_.back().get_hash(), transactions);
  new_block.mine_block(difficulty_);
  chain_.push_back(new_block);
}

const std::vector<Block>& Blockchain::get_chain() const {
  return chain_;
}
