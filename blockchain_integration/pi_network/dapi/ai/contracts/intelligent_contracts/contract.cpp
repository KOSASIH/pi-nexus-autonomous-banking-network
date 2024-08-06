#include "contract.h"
#include <iostream>
#include <libp2p/crypto/key_generator.h>
#include <libp2p/crypto/hash_generator.h>
#include <libp2p/storage/storage_impl.h>

Contract::Contract(const std::string& contract_id, const std::string& code)
    : contract_id_(contract_id), code_(code) {
  // Initialize the private key and hash function
  libp2p::crypto::KeyGenerator key_gen;
  private_key_ = key_gen.generate_key();
  libp2p::crypto::HashGenerator hash_gen;
  hash_function_ = hash_gen.generate_hash();

  // Initialize the storage
  storage_ = libp2p::storage::StorageImpl::create();

  // Parse the contract code and extract functions
  // TO DO: implement contract code parsing
}

Contract::~Contract() {}

void Contract::deploy(const std::string& deployer_id) {
  // Deploy the contract to the blockchain
  // TO DO: implement blockchain deployment
}

std::string Contract::execute(const std::string& function_name, const std::vector<std::string>& args) {
  // Check if the function exists
  if (functions_.find(function_name) != functions_.end()) {
    // Execute the function
    std::string result = functions_[function_name](args);
    return result;
  } else {
    throw std::runtime_error("Function not found");
  }
}

std::string Contract::get_id() const {
  return contract_id_;
}

std::string Contract::get_code() const {
  return code_;
}

libp2p::storage::Storage Contract::get_storage() const {
  return storage_;
}
