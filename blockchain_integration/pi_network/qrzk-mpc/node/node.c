// node.c

#include "node.h"
#include <openssl/aes.h>
#include <openssl/rsa.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Function to create a new node
Node* node_new(uint64_t id, RSA* public_key, RSA* private_key, uint64_t network_id) {
    Node* node = malloc(sizeof(Node));
    node->id = id;
    node->public_key = public_key;
    node->private_key = private_key;
    node->network_id = network_id;
    node->peers = hashmap_new();
    node->blockchain = malloc(sizeof(Block*));
    node->blockchain_size = 0;
    node->mempool = malloc(sizeof(Transaction*));
    node->mempool_size = 0;
    node->node_state = NODE_STATE_IDLE;
    return node;
}

// Function to add a peer to the node
void node_add_peer(Node* node, uint64_t peer_id, int socket) {
    hashmap_put(node->peers, peer_id, socket);
}

// Function to remove a peer from the node
void node_remove_peer(Node* node, uint64_t peer_id) {
    hashmap_remove(node->peers, peer_id);
}

// Function to send a message to a peer
void node_send_message(Node* node, uint64_t peer_id, char* message) {
    int socket = hashmap_get(node->peers, peer_id);
    if (socket!= -1) {
        AES_KEY aes_key;
        AES_set_encrypt_key(node->public_key, 256, &aes_key);
        uint8_t encrypted_message[1024];
        int encrypted_length = AES_encrypt(message, encrypted_message, &aes_key);
        send(socket, encrypted_message, encrypted_length, 0);
    }
}

// Function to receive a message from a peer
void node_receive_message(Node* node, char* message) {
    AES_KEY aes_key;
    AES_set_decrypt_key(node->private_key, 256, &aes_key);
    uint8_t decrypted_message[1024];
    int decrypted_length = AES_decrypt(message, decrypted_message, &aes_key);
    if (decrypted_length > 0) {
        if (strncmp(decrypted_message, "block", 5) == 0) {
            // Handle block message
            Block* block = malloc(sizeof(Block));
            memcpy(block, decrypted_message + 5, decrypted_length - 5);
            node->blockchain = realloc(node->blockchain, (node->blockchain_size + 1) * sizeof(Block));
            node->blockchain[node->blockchain_size] = block;
            node->blockchain_size++;
        } else if (strncmp(decrypted_message, "transaction", 12) == 0) {
            // Handle transaction message
            Transaction* transaction = malloc(sizeof(Transaction));
            memcpy(transaction, decrypted_message + 12, decrypted_length - 12);
            node->mempool = realloc(node->mempool, (node->mempool_size + 1) * sizeof(Transaction));
            node->mempool[node->mempool_size] = transaction;
            node->mempool_size++;
        } else {
            // Handle unknown message
            printf("Unknown message: %s\n", decrypted_message);
        }
    }
}

// Function to start the node
void node_start(Node* node) {
    int socket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(8080);
    inet_pton(AF_INET, "0.0.0.0", &server_address.sin_addr);
    bind(socket, (struct sockaddr*)&server_address, sizeof(server_address));
    listen(socket, 3);

    while (1) {
        struct sockaddr_in client_address;
        socklen_t client_length = sizeof(client_address);
        int client_socket = accept(socket, (struct sockaddr*)&client_address, &client_length);
        if (client_socket!= -1) {
            // Handle incoming connection
            node_add_peer(node, client_socket, client_socket);
            char message[1024];
            recv(client_socket, message, 1024, 0);
            node_receive_message(node, message);
        }
    }
}

int main() {
    // Create a new node
    Node* node = node_new(1, RSA_generate_keypair(2048), RSA_generate_keypair(2048), 1);

    // Start the node
    node_start(node);

    return 0;
}
