// BlockchainTransaction.java
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class BlockchainTransaction {
  private String transactionId;
  private String sender;
  private String recipient;
  private double amount;
  private String timestamp;
  private String previousHash;
  private String hash;

  public BlockchainTransaction(
      String sender, String recipient, double amount, String timestamp, String previousHash) {
    this.sender = sender;
    this.recipient = recipient;
    this.amount = amount;
    this.timestamp = timestamp;
    this.previousHash = previousHash;
    this.hash = calculateHash();
  }

  private String calculateHash() {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      String data = sender + recipient + amount + timestamp + previousHash;
      byte[] bytes = md.digest(data.getBytes());
      StringBuilder sb = new StringBuilder();
      for (byte b : bytes) {
        sb.append(String.format("%02x", b));
      }
      return sb.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }
}
