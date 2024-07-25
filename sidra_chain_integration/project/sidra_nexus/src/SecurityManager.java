package com.sidra.nexus;

import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;

public class SecurityManager {
    private KeyPair keyPair;

    public SecurityManager() {
        // Generate a key pair for encryption and decryption
        KeyPairGenerator kpg = KeyPairGenerator.getInstance("RSA");
        kpg.initialize(2048);
        keyPair = kpg.generateKeyPair();
    }

    public PublicKey getPublicKey() {
        return keyPair.getPublic();
    }

    public PrivateKey getPrivateKey() {
        return keyPair.getPrivate();
    }
}
