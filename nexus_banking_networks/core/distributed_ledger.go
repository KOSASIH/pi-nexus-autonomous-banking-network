package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// Define the block structure
type Block struct {
	Timestamp    int64
	Transactions []Transaction
	Hash         string
	PrevHash     string
}

// Define the transaction structure
type Transaction struct {
	Sender    string
	Recipient string
	Amount    int
}

// Create a new block
func NewBlock(transactions []Transaction, prevHash string) *Block {
	block := &Block{
		Timestamp:    time.Now().Unix(),
		Transactions: transactions,
		Hash:         "",
		PrevHash:     prevHash,
	}
	block.Hash = calculateHash(block)
	return block
}

// Calculate the hash of a block
func calculateHash(block *Block) string {
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("%d%s%s", block.Timestamp, block.Transactions, block.PrevHash)))
	return hex.EncodeToString(hash.Sum(nil))
}

// Create a new blockchain
func NewBlockchain() *Blockchain {
	return &Blockchain{
		chain: []*Block{},
	}
}

// Add a new block to the blockchain
func (bc *Blockchain) AddBlock(transactions []Transaction) {
	prevHash := ""
	if len(bc.chain) > 0 {
		prevHash = bc.chain[len(bc.chain)-1].Hash
	}
	newBlock := NewBlock(transactions, prevHash)
	bc.chain = append(bc.chain, newBlock)
}

// Example usage
func main() {
	bc := NewBlockchain()
	bc.AddBlock([]Transaction{{"Alice", "Bob", 10}, {"Bob", "Charlie", 20}})
	bc.AddBlock([]Transaction{{"Charlie", "David", 30}, {"David", "Alice", 40}})
	fmt.Println("Blockchain:", bc.chain)
}
