#include "ipfs.h"
#include <fstream>
#include <iostream>
#include <libp2p/crypto/key_generator.h>
#include <libp2p/crypto/hash_generator.h>

IPFS::IPFS(const std::string& repo_path) : repo_path_(repo_path) {
  // Initialize the private key and hash function
  libp2p::crypto::KeyGenerator key_gen;
  private_key_ = key_gen.generate_key();
  libp2p::crypto::HashGenerator hash_gen;
  hash_function_ = hash_gen.generate_hash();
}

IPFS::~IPFS() {}

void IPFS::add_file(const std::string& file_path) {
  // Read the file contents
  std::ifstream file(file_path, std::ios::binary);
  std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  // Calculate the CID of the file
  libp2p::crypto::Hash file_hash = hash_function_(file_contents);
  std::string cid = file_hash.to_string();

  // Store the file in the repository
  std::string file_path_in_repo = repo_path_ + "/" + cid;
  std::ofstream file_in_repo(file_path_in_repo, std::ios::binary);
  file_in_repo.write(file_contents.c_str(), file_contents.size());

  // Add the file to the file map
  file_map_[cid] = file_path_in_repo;
}

std::string IPFS::get_file(const std::string& cid) {
  // Check if the file is in the file map
  if (file_map_.find(cid) != file_map_.end()) {
    // Read the file from the repository
    std::ifstream file(file_map_[cid], std::ios::binary);
    std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return file_contents;
  } else {
    throw std::runtime_error("File not found");
  }
}

void IPFS::pin_file(const std::string& cid) {
  // Pin the file to the repository
  // TO DO: implement pinning mechanism
}

void IPFS::unpin_file(const std::string& cid) {
  // Unpin the file from the repository
  // TO DO: implement unpinning mechanism
}

std::vector<std::string> IPFS::list_files() {
  // List all files in the repository
  std::vector<std::string> files;
  for (const auto& file : file_map_) {
    files.push_back(file.first);
  }
  return files;
}
