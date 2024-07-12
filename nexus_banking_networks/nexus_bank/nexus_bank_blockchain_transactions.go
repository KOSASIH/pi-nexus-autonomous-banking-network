package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"time"

	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

type BlockchainTransaction struct {
	TransactionID string
	Amount       int
	Timestamp    time.Time
	Sender       string
	Receiver     string
}

func (t *BlockchainTransaction) Serialize() []byte {
	var buffer bytes.Buffer
	buffer.WriteString(t.TransactionID)
	buffer.WriteString("|")
	buffer.WriteString(fmt.Sprintf("%d", t.Amount))
	buffer.WriteString("|")
	buffer.WriteString(t.Timestamp.String())
	buffer.WriteString("|")
	buffer.WriteString(t.Sender)
	buffer.WriteString("|")
	buffer.WriteString(t.Receiver)
	return buffer.Bytes()
}

func (t *BlockchainTransaction) Hash() string {
	hash := sha256.New()
	hash.Write(t.Serialize())
	return hex.EncodeToString(hash.Sum(nil))
}

func main() {
	// Initialize the chaincode
	chaincode, err := shim.NewChaincode("nexus_bank_blockchain_transactions", nil)
	if err != nil {
		log.Fatal(err)
	}

	// Initialize the ledger
	ledger, err := chaincode.GetStub()
	if err != nil {
		log.Fatal(err)
	}

	// Create a new transaction
	transaction := &BlockchainTransaction{
		TransactionID: "1",
		Amount:       100,
		Timestamp:    time.Now(),
		Sender:       "Alice",
		Receiver:     "Bob",
	}

	// Hash the transaction
	transactionHash := transaction.Hash()

	// Add the transaction to the ledger
	ledger.PutState(transactionHash, []byte(transaction.Serialize()))

	// Get the transaction from the ledger
	transactionBytes, err := ledger.GetState(transactionHash)
	if err != nil {
		log.Fatal(err)
	}

	// Deserialize the transaction
	deserializedTransaction := &BlockchainTransaction{}
	buffer := bytes.NewBuffer(transactionBytes)
	buffer.ReadString('|')
	buffer.ReadString('|')
	buffer.ReadString('|')
	buffer.ReadString('|')
	transactionID := buffer.String()
	buffer.ReadString('|')
	amount := buffer.String()
	buffer.ReadString('|')
	timestamp := buffer.String()
	buffer.ReadString('|')
	sender := buffer.String()
	buffer.ReadString('|')
	receiver := buffer.String()
	deserializedTransaction.TransactionID = transactionID
	deserializedTransaction.Amount = int(amount)
	deserializedTransaction.Timestamp, _ = time.Parse(time.RFC3339, timestamp)
	deserializedTransaction.Sender = sender
	deserializedTransaction.Receiver = receiver

	// Verify the transaction
	if deserializedTransaction.Hash() != transactionHash {
		log.Fatal("Transaction verification failed")
	}

	fmt.Println("Transaction added to the ledger:", deserializedTransaction)
}
