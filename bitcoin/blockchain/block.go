package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"strconv"
	"time"
)

type Block struct {
	Index     int
	Timestamp string
	Data      string
	Hash      string
	PrevHash  string
}

func NewBlock(index int, data string, prevHash string) *Block {
	block := &Block{
		Index:     index,
		Timestamp: time.Now().String(),
		Data:      data,
		PrevHash:  prevHash,
	}
	block.Hash = block.CalculateHash()
	return block
}

func (b *Block) CalculateHash() string {
	data := fmt.Sprintf("%d%s%s%s", b.Index, b.Timestamp, b.Data, b.PrevHash)
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash)
}

func (b *Block) IsValid() bool {
	if b.PrevHash != "" {
		prevBlock := GetBlock(b.Index - 1)
		if prevBlock.Hash != b.PrevHash {
			return false
		}
	}
	return b.Hash == b.CalculateHash()
}
