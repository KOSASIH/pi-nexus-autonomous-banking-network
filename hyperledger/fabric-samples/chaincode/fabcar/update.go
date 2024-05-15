package main

import (
	"bytes"
	"strconv"

	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

func (t *SimpleAsset) CreateCar(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	// TODO: Implement create car logic here
	return shim.Success(nil)
}

func (t *SimpleAsset) ChangeCarOwner(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	// TODO: Implement change car owner logic here
	return shim.Success(nil)
}
