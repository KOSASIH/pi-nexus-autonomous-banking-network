#include "node.h"
#include <iostream>
#include <libp2p/crypto/key_generator.h>
#include <libp2p/crypto/hash_generator.h>
#include <libp2p/network/socket_impl.h>

Node::Node(const std::string& node_id, const std::string& listen_addr)
    : node_id_(node_id), listen_addr_(listen_addr) {
  // Initialize the private key and hash function
  libp2p::crypto::KeyGenerator key_gen;
  private_key_ = key_gen.generate_key();
  libp2p::crypto::HashGenerator hash_gen;
  hash_function_ = hash_gen.generate_hash();

  // Initialize the socket
  socket_ = libp2p::network::SocketImpl::create(listen_addr_);
}

Node::~Node() {
  // Close the socket
  socket_.close();
}

void Node::start() {
  // Start listening on the socket
  socket_.listen();
}

void Node::stop() {
  // Stop listening on the socket
  socket_.stop();
}

void Node::connect(const std::string& node_id, const std::string& addr) {
  // Create a new socket connection
  libp2p::network::Socket connection = libp2p::network::SocketImpl::create(addr);
  connections_[node_id] = connection;
}

void Node::disconnect(const std::string& node_id) {
  // Close the socket connection
  connections_[node_id].close();
  connections_.erase(node_id);
}

void Node::send_message(const std::string& node_id, const std::string& message) {
  // Get the socket connection
  libp2p::network::Socket connection = connections_[node_id];

  // Encrypt the message using the private key
  std::string encrypted_message = private_key_.encrypt(message);

  // Send the encrypted message over the socket
  connection.send(encrypted_message);
}

void Node::receive_message(const std::string& node_id, const std::string& message) {
  // Get the socket connection
  libp2p::network::Socket connection = connections_[node_id];

  // Decrypt the message using the private key
  std::string decrypted_message = private_key_.decrypt(message);

  // Process the decrypted message
  std::cout << "Received message from " << node_id << ": " << decrypted_message << std::endl;
}

std::string Node::get_id() const {
  return node_id_;
}

std::string Node::get_listen_addr() const {
  return listen_addr_;
}
