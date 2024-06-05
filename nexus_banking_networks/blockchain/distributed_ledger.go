// distributed_ledger.go
package main

import (
	"fmt"
	"sync"

	"github.com/hyperledger/fabric-sdk-go/pkg/fabric-sdk-go"
)

type DistributedLedger struct {
	sdk *fabric_sdk_go.FabricSDK
}

func NewDistributedLedger() *DistributedLedger {
	sdk, err := fabric_sdk_go.NewFabricSDK()
	if err != nil {
		fmt.Println(err)
		return nil
	}
	return &DistributedLedger{sdk: sdk}
}

func (dl *DistributedLedger) CreateChannel(channelName string) error {
	// Create a new channel on the distributed ledger
	channel, err := dl.sdk.Channel(channelName)
	if err != nil {
		return err
	}
	return channel.Create()
}

func (dl *DistributedLedger) JoinChannel(channelName string, peerURL string) error {
	// Join a peer to the channel on the distributed ledger
	channel, err := dl.sdk.Channel(channelName)
	if err != nil {
		return err
	}
	return channel.Join(peerURL)
}

func (dl *DistributedLedger) InvokeChaincode(chaincodeName string, args []string) ([]byte, error) {
	// Invoke a chaincode on the distributed ledger
	channel, err := dl.sdk.Channel("mychannel")
	if err != nil {
		return nil, err
	}
	return channel.Invoke(chaincodeName, args)
}

func main() {
	dl := NewDistributedLedger()
	err := dl.CreateChannel("mychannel")
	if err != nil {
		fmt.Println(err)
		return
	}
	err = dl.JoinChannel("mychannel", "grpc://localhost:7051")
	if err != nil {
		fmt.Println(err)
		return
	}
	result, err := dl.InvokeChaincode("mychaincode", []string{"init", "100"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(result))
}
