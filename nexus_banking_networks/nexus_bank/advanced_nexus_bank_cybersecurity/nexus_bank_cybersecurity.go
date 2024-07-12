package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"
)

func encrypt(plaintext []byte, key []byte) ([]byte, error) {
	// Create a new AES cipher
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Generate a random IV
	iv := make([]byte, c.BlockSize())
	_, err = io.ReadFull(rand.Reader, iv)
	if err != nil {
		return nil, err
	}

	// Encrypt the plaintext
	cfb := cipher.NewCFBEncrypter(c, iv)
	ciphertext := make([]byte, len(plaintext))
	cfb.XORKeyStream(ciphertext, plaintext)

	// Return the IV and ciphertext
	return append(iv, ciphertext...), nil
}

func decrypt(ciphertext []byte, key []byte) ([]byte, error) {
	// Extract the IV from the ciphertext
	iv := ciphertext[:aes.BlockSize]
	ciphertext = ciphertext[aes.BlockSize:]

	// Create a new AES cipher
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Decrypt the ciphertext
	cfb := cipher.NewCFBDecrypter(c, iv)
	plaintext := make([]byte, len(ciphertext))
	cfb.XORKeyStream(plaintext, ciphertext)

	return plaintext, nil
}

func main() {
	// Generate a random key
	key := make([]byte, 32)
	_, err := io.ReadFull(rand.Reader, key)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Encrypt a message
	plaintext := []byte("Hello, World!")
	ciphertext, err := encrypt(plaintext, key)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Decrypt the message
	decrypted, err := decrypt(ciphertext, key)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(string(decrypted))
}
