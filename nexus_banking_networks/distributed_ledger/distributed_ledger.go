// distributed_ledger.go
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

type Block struct {
	Timestamp    time.Time
	Transactions []Transaction
	Hash         string
	PrevHash     string
}

type Transaction struct {
	Sender    string
	Recipient string
	Amount    float64
}

func calculateHash(block *Block) string {
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("%v%v", block.Timestamp, block.Transactions)))
	return hex.EncodeToString(hash.Sum(nil))
}

func createNewBlock(transactions []Transaction, prevHash string) *Block {
	block := &Block{
		Timestamp:    time.Now(),
		Transactions: transactions,
		Hash:         calculateHash(block),
		PrevHash:     prevHash,
	}
	return block
}

func main() {
	// Create a new blockchain with a genesis block
	genesisBlock := createNewBlock([]Transaction{}, "")
	blockchain := []*Block{genesisBlock}

	// Add new blocks to the blockchain
	for i := 0; i < 10; i++ {
		transactions := []Transaction{
			{Sender: "Alice", Recipient: "Bob", Amount: 10.0},
			{Sender: "Bob", Recipient: "Charlie", Amount: 5.0},
		}
		newBlock := createNewBlock(transactions, blockchain[i].Hash)
		blockchain = append(blockchain, newBlock)
	}

	// Print the blockchain
	for _, block := range blockchain {
		fmt.Printf("Block %v: %v\n", block.Timestamp, block.Hash)
	}
}
