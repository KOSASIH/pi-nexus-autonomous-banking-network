#include <iostream>
#include <string>
#include <vector>
#include <bitcoin/bitcoin.h>

class TransactionLedger {
public:
    TransactionLedger() : blockchain_(new BC::Blockchain()) {}

    void addTransaction(const std::string& transaction) {
        // Create a new block and add the transaction
        BC::Block block;
        block.addTransaction(transaction);
        blockchain_->addBlock(block);
    }

    std::string getTransactionHistory(const std::string& user_id) {
        // Retrieve the transaction history for the given user_id
        std::vector<std::string> transactions;
        blockchain_->forEachBlock([&](const BC::Block& block) {
            for (const auto& tx : block.getTransactions()) {
                if (tx.getUserID() == user_id) {
                    transactions.push_back(tx.toString());
                }
            }
        });
        return join(transactions, "\n");
    }

private:
    std::unique_ptr<BC::Blockchain> blockchain_;
};

int main() {
    TransactionLedger ledger;

    // Add some sample transactions
    ledger.addTransaction("user1,100,withdrawal");
    ledger.addTransaction("user2,50,deposit");
    ledger.addTransaction("user1,200,transfer");

    // Retrieve the transaction history for user1
    std::string history = ledger.getTransactionHistory("user1");
    std::cout << "Transaction history for user1:\n" << history << std::endl;

    return 0;
}
