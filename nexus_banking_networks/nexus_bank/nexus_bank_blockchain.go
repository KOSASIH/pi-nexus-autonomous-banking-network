package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"github.com/btcsuite/btcd/btcec"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/txscript"
	"github.com/btcsuite/btcd/wire"
)

type Blockchain struct {
	chaincfg *chaincfg.Params
}

func NewBlockchain() *Blockchain {
	return &Blockchain{
		chaincfg: &chaincfg.MainNetParams,
	}
}

func (b *Blockchain) CreateTransaction(from, to string, amount int64) (*wire.MsgTx, error) {
	// Create a new transaction
	tx := wire.NewMsgTx(wire.TxVersion)

	// Add inputs
	tx.AddTxIn(wire.NewTxIn(wire.NewOutPoint(from, 0), nil, nil))

	// Add outputs
	tx.AddTxOut(wire.NewTxOut(amount, to))

	// Sign the transaction
	privKey, err := ecdsa.GenerateKey(btcec.S256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	sig, err := txscript.SignTxOutput(b.chaincfg, tx, 0, privKey, txscript.SigHashAll, true)
	if err != nil {
		return nil, err
	}
	tx.TxIn[0].SignatureScript = sig

	return tx, nil
}

func main() {
	b := NewBlockchain()
	tx, err := b.CreateTransaction("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1B1tP1eP5QGefi2DMPTfTL5SLmv7DivfNb", 1000000)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tx)
}
