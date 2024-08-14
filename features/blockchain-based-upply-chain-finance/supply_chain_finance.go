package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"strings"
	"time"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SupplyChainFinance is the main struct for the supply chain finance smart contract
type SupplyChainFinance struct {
	contractapi.Contract
}

// Trade represents a trade in the supply chain
type Trade struct {
	ID          string `json:"id"`
	Buyer       string `json:"buyer"`
	Seller      string `json:"seller"`
	Product     string `json:"product"`
	Quantity    int    `json:"quantity"`
	Price       *big.Rat `json:"price"`
	Status      string `json:"status"`
	Timestamp   int64  `json:"timestamp"`
	Hash        string `json:"hash"`
	Signatures  []string `json:"signatures"`
}

// Ledger represents the ledger of trades
type Ledger struct {
	Trades []Trade `json:"trades"`
}

// Init initializes the supply chain finance smart contract
func (s *SupplyChainFinance) Init(stub shim.ChaincodeStubInterface) []byte {
	log.Println("Initializing supply chain finance smart contract")
	return nil
}

// Invoke handles incoming requests to the supply chain finance smart contract
func (s *SupplyChainFinance) Invoke(stub shim.ChaincodeStubInterface) ([]byte, error) {
	log.Println("Received invoke request")
	function, args := stub.GetFunctionAndParameters()
	switch function {
	case "createTrade":
		return s.createTrade(stub, args)
	case "updateTrade":
		return s.updateTrade(stub, args)
	case "getTrade":
		return s.getTrade(stub, args)
	case "getAllTrades":
		return s.getAllTrades(stub, args)
	case "getTradeHistory":
		return s.getTradeHistory(stub, args)
	case "verifyTrade":
		return s.verifyTrade(stub, args)
	default:
		return nil, fmt.Errorf("Invalid function name: %s", function)
	}
}

// createTrade creates a new trade in the supply chain
func (s *SupplyChainFinance) createTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Creating new trade")
	if len(args) != 6 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 6, got %d", len(args))
	}

	trade := Trade{
		ID:          args[0],
		Buyer:       args[1],
		Seller:      args[2],
		Product:     args[3],
		Quantity:    parseInt(args[4]),
		Price:       new(big.Rat).SetString(args[5]),
		Status:      "pending",
		Timestamp:   time.Now().Unix(),
		Hash:        calculateHash(trade),
		Signatures:  []string{},
	}

	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	ledger.Trades = append(ledger.Trades, trade)

	err = stub.PutState("ledger", ledger)
	if err != nil {
		return nil, err
	}

	return []byte("Trade created successfully"), nil
}

// updateTrade updates an existing trade in the supply chain
func (s *SupplyChainFinance) updateTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Updating trade")
	if len(args) != 2 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 2, got %d", len(args))
	}

	tradeID := args[0]
	status := args[1]

	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	for i, trade := range ledger.Trades {
		if trade.ID == tradeID {
			trade.Status = status
			trade.Timestamp = time.Now().Unix()
			trade.Hash = calculateHash(trade)
			ledger.Trades[i] = trade
			break
		}
	}

	err = stub.PutState("ledger", ledger)
	if err != nil {
		return nil, err
	}

	return []byte("Trade updated successfully"), nil
}

// getTrade retrieves a trade from the supply chain
func (s *SupplyChainFinance) getTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Retrieving trade")
	if len(args) != 1 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 1, got %d", len(args
