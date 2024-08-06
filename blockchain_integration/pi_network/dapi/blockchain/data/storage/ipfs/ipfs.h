#ifndef IPFS_H
#define IPFS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <libp2p/crypto/key.h>
#include <libp2p/crypto/hash.h>

class IPFS {
public:
  IPFS(const std::string& repo_path);
  ~IPFS();

  // Add a file to the IPFS repository
  void add_file(const std::string& file_path);

  // Get a file from the IPFS repository
  std::string get_file(const std::string& cid);

  // Pin a file to the IPFS repository
  void pin_file(const std::string& cid);

  // Unpin a file from the IPFS repository
  void unpin_file(const std::string& cid);

  // Get a list of all files in the IPFS repository
  std::vector<std::string> list_files();

private:
  std::string repo_path_;
  std::unordered_map<std::string, std::string> file_map_;
  libp2p::crypto::Key private_key_;
  libp2p::crypto::Hash hash_function_;
};

#endif  // IPFS_H
