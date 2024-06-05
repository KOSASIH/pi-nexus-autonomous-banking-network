// TransactionVerificationSystem.java
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.util.ArrayList;
import java.util.List;

import org.bouncycastle.jce.provider.BouncyCastleProvider;

public class TransactionVerificationSystem {
    private KeyPair keyPair;
    private List<Transaction> transactionChain;

    public TransactionVerificationSystem() {
        keyPair = generateKeyPair();
        transactionChain = new ArrayList<>();
    }

    private KeyPair generateKeyPair() {
        KeyPairGenerator kpg = KeyPairGenerator.getInstance("RSA", new BouncyCastleProvider());
        kpg.initialize(2048);
        return kpg.generateKeyPair();
    }

    public void addTransaction(Transaction transaction) {
        transaction.signTransaction(keyPair.getPrivate());
        transactionChain.add(transaction);
    }

    public boolean verifyTransactionChain() {
        for (int i = 0; i < transactionChain.size() - 1; i++) {
            Transaction current = transactionChain.get(i);
            Transaction next = transactionChain.get(i + 1);
            if (!current.verifySignature(next.getHash(), keyPair.getPublic())) {
                return false;
            }
        }
        return true;
    }

    public static class Transaction {
        private String hash;
        private String data;
        private byte[] signature;

        public Transaction(String data) {
            this.data = data;
            this.hash = calculateHash(data);
        }

        public void signTransaction(PrivateKey privateKey) {
            signature = signData(hash, privateKey);
        }

        public boolean verifySignature(String hash, PublicKey publicKey) {
            return verifyData(hash, signature, publicKey);
        }

        // Implementation of hash calculation, signing, and verification using Bouncy Castle
    }
}

// Example usage:
TransactionVerificationSystem tvs = new TransactionVerificationSystem();
Transaction tx1 = new Transaction("Transaction 1 data");
tvs.addTransaction(tx1);
Transaction tx2 = new Transaction("Transaction 2 data");
tvs.addTransaction(tx2);
System.out.println("Transaction chain is valid: " + tvs.verifyTransactionChain());
