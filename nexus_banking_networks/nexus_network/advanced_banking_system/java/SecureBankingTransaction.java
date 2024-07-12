import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import javax.crypto.Cipher;

public class SecureBankingTransaction {
  public static void main(String[] args) throws Exception {
    KeyPairGenerator kpg = KeyPairGenerator.getInstance("RSA");
    kpg.initialize(2048);
    KeyPair kp = kpg.generateKeyPair();
    PrivateKey privateKey = kp.getPrivate();
    PublicKey publicKey = kp.getPublic();

    String transactionData = "Transaction data";
    Cipher cipher = Cipher.getInstance("RSA");
    cipher.init(Cipher.ENCRYPT_MODE, publicKey);
    byte[] encryptedData = cipher.doFinal(transactionData.getBytes());

    cipher.init(Cipher.DECRYPT_MODE, privateKey);
    byte[] decryptedData = cipher.doFinal(encryptedData);
    System.out.println("Decrypted data: " + new String(decryptedData));
  }
}
