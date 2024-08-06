#include "node.h"

int main() {
  Node node("node1", "localhost:8080");
  node.init();

  // Register other nodes
  node.register_node("node2", "localhost:8081");
  node.register_node("node3", "localhost:8082");

  // Send messages to other nodes
  node.handle_message("Hello, node2!", "node2");
  node.handle_message("Hello, node3!", "node3");

  // Shutdown the node
  node.shutdown();

  return 0;
}
