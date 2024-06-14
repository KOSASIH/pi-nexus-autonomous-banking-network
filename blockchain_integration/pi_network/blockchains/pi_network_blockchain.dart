// pi_network_blockchain.dart
import 'dart:convert';
import 'dart:math';

class PiNetworkBlockchain {
  List<Block> chain = [];
  List<Transaction> pendingTransactions = [];
  String minerAddress = "";

  PiNetworkBlockchain() {
    // Initialize blockchain with genesis block
  }

  void createNewBlock(String minerAddress) {
    // Create new block and add to chain
  }

  void addTransaction(Transaction tx) {
    // Add transaction to pending transactions
  }

  void minePendingTransactions(String minerAddress) {
    // Mine pending transactions and create new block
  }

  double getBalanceOfAddress(String address) {
    // Return balance of address
  }

  Transaction getTransactionByID(String txID) {
    // Return transaction by ID
  }
}

class Block {
  int index;
  DateTime timestamp;
  List<Transaction> transactions;
  String hash;
  String prevHash;

  Block(
      {required this.index,
      required this.timestamp,
      required this.transactions,
      required this.hash,
      required this.prevHash});

  factory Block.fromJson(Map<String, dynamic> json) {
    return Block(
        index: json['index'],
        timestamp: DateTime.parse(json['timestamp']),
        transactions: List<Transaction>.from(
            json['transactions'].map((tx) => Transaction.fromJson(tx))),
        hash: json['hash'],
        prevHash: json['prevHash']);
  }

  Map<String, dynamic> toJson() => {
        'index': index,
        'timestamp': timestamp.toIso8601String(),
        'transactions': transactions.map((tx) => tx.toJson()),
        'hash': hash,
        'prevHash': prevHash,
      };
}

class Transaction {
  String sender;
  String recipient;
  double amount;

  Transaction({required this.sender, required this.recipient, required this.amount});

  factory Transaction.fromJson(Map<String, dynamic> json) {
    return Transaction(
        sender: json['sender'],
        recipient: json['recipient'],
        amount: json['amount']);
  }

  Map<String, dynamic> toJson() => {
        'sender': sender,
        'recipient': recipient,
        'amount': amount,
      };
}
