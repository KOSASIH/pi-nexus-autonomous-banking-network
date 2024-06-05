// TransactionLedger.java
import java.util.ArrayList;
import java.util.List;

public class TransactionLedger {
    private List<Transaction> transactions;

    public TransactionLedger() {
        this.transactions = new ArrayList<>();
    }

    public void addTransaction(Transaction transaction) {
        transactions.add(transaction);
        // Update blockchain
        updateBlockchain();
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }

    private void updateBlockchain() {
        // Implement blockchain update logic here
        System.out.println("Blockchain updated!");
    }
}

class Transaction {
    private String id;
    private double amount;
    private String category;

    public Transaction(String id, double amount, String category) {
        this.id = id;
        this.amount = amount;
        this.category = category;
    }

    public String getId() {
        return id;
    }

    public double getAmount() {
        return amount;
    }

    public String getCategory() {
        return category;
    }
}

// Example usage:
TransactionLedger ledger = new TransactionLedger();
Transaction transaction = new Transaction("TX123", 500.0, "deposit");
ledger.addTransaction(transaction);
System.out.println("Transactions:");
for (Transaction tx : ledger.getTransactions()) {
    System.out.println(tx.getId() + ": " + tx.getAmount() + " (" + tx.getCategory() + ")");
}
