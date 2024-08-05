package wallet

import (
	"crypto/ecdsa"
	"crypto/rand"
)

type PiWallet struct {
	//...
}

func (pw *PiWallet) GenerateKeyPair() (*ecdsa.PrivateKey, *ecdsa.PublicKey) {
	//...
}

func (pw *PiWallet) CreateTransaction(recipient string, amount int) *Transaction {
	//...
}

func (pw *PiWallet) SignTransaction(tx *Transaction) []byte {
	//...
}
