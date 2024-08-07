package diam

import (
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
)

type Identity struct {
	privateKey *ecdsa.PrivateKey
	publicKey  *ecdsa.PublicKey
}

func NewIdentity() (*Identity, error) {
	privateKey, err := ecdsa.GenerateKey(rand.Reader, 256)
	if err != nil {
		return nil, err
	}
	publicKey := privateKey.Public()

	return &Identity{privateKey, publicKey}, nil
}

func (i *Identity) Sign(data []byte) ([]byte, error) {
	hash := sha256.Sum256(data)
	signature, err := i.privateKey.Sign(rand.Reader, hash[:], nil)
	if err != nil {
		return nil, err
	}
	return signature, nil
}
