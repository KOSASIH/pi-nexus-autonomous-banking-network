// sidra_data_privacy_platform/src/main/java/com/example/SidraDataPrivacyPlatform.java
import homomorphic_encryption.HomomorphicEncryption;
import secure_multi_party_computation.SecureMultiPartyComputation;

public class SidraDataPrivacyPlatform {
    private HomomorphicEncryption homomorphicEncryption;
    private SecureMultiPartyComputation secureMultiPartyComputation;

    public SidraDataPrivacyPlatform() {
        homomorphicEncryption = new HomomorphicEncryption();
        secureMultiPartyComputation = new SecureMultiPartyComputation();
    }

    public byte[] encryptData(byte[] data) {
        return homomorphicEncryption.encrypt(data);
    }

    public byte[] decryptData(byte[] encryptedData) {
        return homomorphicEncryption.decrypt(encryptedData);
    }

    public byte[] computeOnEncryptedData(byte[] encryptedData1, byte[] encryptedData2) {
        return secureMultiPartyComputation.compute(encryptedData1, encryptedData2);
    }
}
