#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Node {
public:
  Node(const std::string& node_id, const std::string& node_endpoint);
  ~Node();

  // Node initialization
  void init();

  // Node shutdown
  void shutdown();

  // Message handling
  void handle_message(const std::string& message_data, const std::string& source_node_id);

  // Node registration
  void register_node(const std::string& node_id, const std::string& node_endpoint);

  // Node deregistration
  void deregister_node(const std::string& node_id);

  // Node discovery
  void discover_nodes();

private:
  std::string node_id_;
  std::string node_endpoint_;
  std::unordered_map<std::string, std::string> node_endpoints_;
  std::vector<std::thread> worker_threads_;
  std::mutex message_mutex_;
  std::condition_variable message_cv_;
  bool shutdown_requested_;
};

#endif  // NODE_H
