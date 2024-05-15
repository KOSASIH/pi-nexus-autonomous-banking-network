package main

import (
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

type SimpleAsset struct {
}

func (t *SimpleAsset) Init(stub shim.ChaincodeStubInterface) pb.Response {
	// TODO: Implement initialization logic here
	return shim.Success(nil)
}

func (t *SimpleAsset) Invoke(stub shim.ChaincodeStubInterface) pb.Response {
	// TODO: Implement transaction logic here
	return shim.Success(nil)
}

func main() {
	err := shim.Start(new(SimpleAsset))
	if err != nil {
		fmt.Printf("Error starting SimpleAsset chaincode: %s", err)
	}
}
