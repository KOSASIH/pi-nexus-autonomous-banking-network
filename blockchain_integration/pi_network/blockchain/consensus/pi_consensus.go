package consensus

import (
	"crypto/sha256"
	"encoding/hex"
	"math/big"
	"time"
)

type PiConsensus struct {
	//...
}

func (pc *PiConsensus) VerifyBlock(block *Block) bool {
	//...
}

func (pc *PiConsensus) CalculateHash(block *Block) string {
	//...
}

func (pc *PiConsensus) ValidateTransaction(tx *Transaction) bool {
	//...
}
