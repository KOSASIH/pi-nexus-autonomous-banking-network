package com.sidra;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

public class Cybersecurity {
    public static void main(String[] args) {
        // Set up a secret key
        SecretKeySpec secretKey = new SecretKeySpec("your_secret_key".getBytes(), "AES");

        // Encrypt a message
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encryptedMessage = cipher.doFinal("Hello, world!".getBytes());
    }
}
