// piNodeSecurity.go
package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type PiNodeSecurity struct {
	nodeAddress common.Address
}

func NewPiNodeSecurity(nodeAddress common.Address) *PiNodeSecurity {
	return &PiNodeSecurity{nodeAddress: nodeAddress}
}

func (pns *PiNodeSecurity) GenerateNodeKey() ([]byte, error) {
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	return privateKey.D.Bytes(), nil
}

func (pns *PiNodeSecurity) SignNodeMessage(message []byte) ([]byte, error) {
	privateKey, err := pns.GenerateNodeKey()
	if err != nil {
		return nil, err
	}
	hash := sha256.Sum256(message)
	signature, err := ecdsa.SignASN1(rand.Reader, privateKey, hash[:])
	if err != nil {
		return nil, err
	}
	return signature, nil
}
