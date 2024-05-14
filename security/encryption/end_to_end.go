package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/subtle"
	"io"
)

type endToEndEncryption struct {
	key []byte
}

func NewEndToEndEncryption(key []byte) *endToEndEncryption {
	return &endToEndEncryption{
		key: key,
	}
}

func (e *endToEndEncryption) Encrypt(plaintext []byte) ([]byte, error) {
	block, err := aes.NewCipher(e.key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
	
