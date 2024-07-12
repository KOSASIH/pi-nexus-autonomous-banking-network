package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type Transaction struct {
	From     string
	To       string
	Amount   int
	Signature []byte
}

func main() {
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		fmt.Println(err)
		return
	}

	tx := Transaction{
		From:     "Alice",
		To:       "Bob",
		Amount:   10,
		Signature: nil,
	}

	hash := sha256.Sum256([]byte(tx.From + tx.To + fmt.Sprintf("%d", tx.Amount)))
	sig, err := ecdsa.Sign(rand.Reader, privKey, hash[:])
	if err != nil {
		fmt.Println(err)
		return
	}

	tx.Signature = sig
	fmt.Println(hex.EncodeToString(tx.Signature))
}
