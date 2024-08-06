#ifndef CONTRACT_H
#define CONTRACT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <libp2p/crypto/key.h>
#include <libp2p/crypto/hash.h>
#include <libp2p/storage/storage.h>

class Contract {
public:
  Contract(const std::string& contract_id, const std::string& code);
  ~Contract();

  // Deploy the contract to the blockchain
  void deploy(const std::string& deployer_id);

  // Execute a function on the contract
  std::string execute(const std::string& function_name, const std::vector<std::string>& args);

  // Get the contract's ID
  std::string get_id() const;

  // Get the contract's code
  std::string get_code() const;

  // Get the contract's storage
  libp2p::storage::Storage get_storage() const;

private:
  std::string contract_id_;
  std::string code_;
  libp2p::crypto::Key private_key_;
  libp2p::crypto::Hash hash_function_;
  libp2p::storage::Storage storage_;
  std::unordered_map<std::string, std::string> functions_;
};

#endif  // CONTRACT_H
