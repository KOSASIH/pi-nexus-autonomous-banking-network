#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <libp2p/crypto/key.h>
#include <libp2p/crypto/hash.h>
#include <libp2p/network/socket.h>

class Node {
public:
  Node(const std::string& node_id, const std::string& listen_addr);
  ~Node();

  // Start the node
  void start();

  // Stop the node
  void stop();

  // Connect to another node
  void connect(const std::string& node_id, const std::string& addr);

  // Disconnect from another node
  void disconnect(const std::string& node_id);

  // Send a message to another node
  void send_message(const std::string& node_id, const std::string& message);

  // Receive a message from another node
  void receive_message(const std::string& node_id, const std::string& message);

  // Get the node's ID
  std::string get_id() const;

  // Get the node's listen address
  std::string get_listen_addr() const;

private:
  std::string node_id_;
  std::string listen_addr_;
  libp2p::crypto::Key private_key_;
  libp2p::crypto::Hash hash_function_;
  libp2p::network::Socket socket_;
  std::unordered_map<std::string, libp2p::network::Socket> connections_;
};

#endif  // NODE_H
