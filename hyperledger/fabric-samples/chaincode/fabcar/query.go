package main

import (
	"bytes"
	"fmt"

	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

func (t *SimpleAsset) Query(stub shim.ChaincodeStubInterface) pb.Response {
	// TODO: Implement query logic here
	return shim.Success(nil)
}
