// piNode.go
package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"

	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

type PiNode struct {
	privateKey *ecdsa.PrivateKey
	nodeAddress common.Address
}

func NewPiNode(privateKeyHex string) (*PiNode, error) {
	privateKey, err := hex.DecodeString(privateKeyHex)
	if err != nil {
		return nil, err
	}
	privateKeyECDSA, err := ecdsa.GenerateKey(ecdsa.S256(), privateKey)
	if err!= nil {
		return nil, err
	}
	nodeAddress := accounts.PrivateKeyToAddress(privateKeyECDSA)
	return &PiNode{privateKey: privateKeyECDSA, nodeAddress: nodeAddress}, nil
}

func (pn *PiNode) Start() error {
	// Initialize node with private key and address
	fmt.Println("Pi Node started:", pn.nodeAddress.Hex())
	return nil
}

func (pn *PiNode) HandleTransaction(tx *types.Transaction) error {
	// Handle incoming transactions, validate, and process
	fmt.Println("Received transaction:", tx.Hash().Hex())
	return nil
}

func main() {
	privateKeyHex := "YOUR_PRIVATE_KEY_HEX"
	pn, err := NewPiNode(privateKeyHex)
	if err!= nil {
		log.Fatal(err)
	}
	pn.Start()
}
