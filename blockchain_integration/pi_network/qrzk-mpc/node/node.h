// node.h

#ifndef NODE_H
#define NODE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/aes.h>
#include <openssl/rsa.h>

// Define the Node struct
typedef struct Node {
    uint64_t id;
    RSA* public_key;
    RSA* private_key;
    uint64_t network_id;
    HashMap* peers;
    Block** blockchain;
    int blockchain_size;
    Transaction** mempool;
    int mempool_size;
    NodeState node_state;
} Node;

// Define the NodeState enum
typedef enum {
    NODE_STATE_IDLE,
    NODE_STATE_SYNCING,
    NODE_STATE_MINING
} NodeState;

// Define the Block struct
typedef struct Block {
    uint64_t id;
    Transaction** transactions;
    int transaction_count;
    uint8_t previous_hash[32];
    uint64_t timestamp;
} Block;

// Define the Transaction struct
typedef struct Transaction {
    uint64_t id;
    uint8_t from[20];
    uint8_t to[20];
    uint64_t amount;
} Transaction;

// Function to create a new node
Node* node_new(uint64_t id, RSA* public_key, RSA* private_key, uint64_t network_id);

// Function to add a peer to the node
void node_add_peer(Node* node, uint64_t peer_id, int socket);

// Function to remove a peer from the node
void node_remove_peer(Node* node, uint64_t peer_id);

// Function to send a message to a peer
void node_send_message(Node* node, uint64_t peer_id, char* message);

// Function to receive a message from a peer
void node_receive_message(Node* node, char* message);

// Function to start the node
void node_start(Node* node);

#endif // NODE_H
