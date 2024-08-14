package main

import (
	"fmt"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
)

type SupplyChainFinance struct {
}

func (s *SupplyChainFinance) Init(stub shim.ChaincodeStubInterface) []byte {
	fmt.Println("Initializing Supply Chain Finance Chaincode")
	return nil
}

func (s *SupplyChainFinance) Invoke(stub shim.ChaincodeStubInterface) ([]byte, error) {
	fmt.Println("Received invoke request")

	// Get the function and args from the stub
	funcName, args := stub.GetFunctionAndParameters()

	// Handle different functions
	switch funcName {
	case "createTrade":
		return s.createTrade(stub, args)
	case "updateTrade":
		return s.updateTrade(stub, args)
	case "getTrade":
		return s.getTrade(stub, args)
	default:
		return nil, errors.New("Invalid function name")
	}
}

func (s *SupplyChainFinance) createTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	// Create a new trade object
	trade := Trade{
		ID:          args[0],
		Buyer:       args[1],
		Seller:      args[2],
		Product:     args[3],
		Quantity:    args[4],
		Price:       args[5],
		Status:      "pending",
	}

	// Put the trade object into the ledger
	err := stub.PutState(trade.ID, trade)
	if err != nil {
		return nil, err
	}

	return []byte("Trade created successfully"), nil
}

func (s *SupplyChainFinance) updateTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	// Get the trade object from the ledger
	trade, err := s.getTrade(stub, args[0])
	if err != nil {
		return nil, err
	}

	// Update the trade object
	trade.Status = args[1]

	// Put the updated trade object into the ledger
	err = stub.PutState(trade.ID, trade)
	if err != nil {
		return nil, err
	}

	return []byte("Trade updated successfully"), nil
}

func (s *SupplyChainFinance) getTrade(stub shim.ChaincodeStubInterface, tradeID string) ([]byte, error) {
	// Get the trade object from the ledger
	trade, err := stub.GetState(tradeID)
	if err != nil {
		return nil, err
	}

	return trade, nil
}

type Trade struct {
	ID          string `json:"id"`
	Buyer       string `json:"buyer"`
	Seller      string `json:"seller"`
	Product     string `json:"product"`
	Quantity    string `json:"quantity"`
	Price       string `json:"price"`
	Status      string `json:"status"`
}
