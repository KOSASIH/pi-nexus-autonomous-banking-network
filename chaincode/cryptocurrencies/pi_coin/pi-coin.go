package main

import (
	"bytes"
	"fmt"

	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

type PiCoin struct{}

func (t *PiCoin) Init(stub shim.ChaincodeStub, function string, args []string) pb.Response {
	return shim.Success(nil)
}

func (t *PiCoin) Invoke(stub shim.ChaincodeStub, function string, args []string) pb.Response {
	switch function {
	case "issue":
		return t.issue(stub, args)
	case "transfer":
		return t.transfer(stub, args)
	default:
		return shim.Error("Invalid function")
	}
}

func (t *PiCoin) issue(stub shim.ChaincodeStub, args []string) pb.Response {
	if len(args) != 2 {
		return shim.Error("Invalid number of arguments")
	}

	// Issue new Pi Coins to an address
	err := stub.PutState(args[0], []byte(args[1]))
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success(nil)
}

func (t *PiCoin) transfer(stub shim.ChaincodeStub, args []string) pb.Response {
	if len(args) != 3 {
		return shim.Error("Invalid number of arguments")
	}

	// Transfer Pi Coins from one address to another
	fromValue, err := stub.GetState(args[0])
	if err != nil {
		return shim.Error(err.Error())
	}

	toValue, err := stub.GetState(args[1])
	if err != nil {
		return shim.Error(err.Error())
	}

	newFromValue := new(string)
	*newFromValue = fmt.Sprintf("%s", fromValue)
	*newFromValue = fmt.Sprintf("%s", strings.TrimSpace(strings.Replace(*newFromValue, args[2], "", 1)))

	newToValue := new(string)
	*newToValue = fmt.Sprintf("%s", toValue)
	*newToValue = fmt.Sprintf("%s%s", strings.TrimSpace(*newToValue), args[2])

	err = stub.PutState(args[0], []byte(*newFromValue))
	if err != nil {
		return shim.Error(err.Error())
	}

	err = stub.PutState(args[1], []byte(*newToValue))
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success(nil)
}
