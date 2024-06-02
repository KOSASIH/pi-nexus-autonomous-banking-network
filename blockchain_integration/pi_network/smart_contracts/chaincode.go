package main

import (
	"fmt"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
)

type PINetworkChaincode struct {
}

func (c *PINetworkChaincode) Init(stub shim.ChaincodeStubInterface) []byte {
	fmt.Println("PINetworkChaincode Init")
	return nil
}

func (c *PINetworkChaincode) Invoke(stub shim.ChaincodeStubInterface) ([]byte, error) {
	fmt.Println("PINetworkChaincode Invoke")
	args := stub.GetArgs()
	if len(args)!= 2 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 2")
	}

	var funcName string
	var params []byte

	funcName = args[0]
	params = args[1]

	if funcName == "transfer" {
		return c.transfer(stub, params)
	} else if funcName == "getBalance" {
		return c.getBalance(stub, params)
	} else {
		return nil, fmt.Errorf("Invalid function name. Expecting 'transfer' or 'getBalance'")
	}
}

func (c *PINetworkChaincode) transfer(stub shim.ChaincodeStubInterface, params []byte) ([]byte, error) {
	fmt.Println("PINetworkChaincode transfer")
	// Implement transfer logic here
	return nil, nil
}

func (c *PINetworkChaincode) getBalance(stub shim.ChaincodeStubInterface, params []byte) ([]byte, error) {
	fmt.Println("PINetworkChaincode getBalance")
	// Implement getBalance logic here
	return nil, nil
}

func main() {
	fmt.Println("PINetworkChaincode main")
}
