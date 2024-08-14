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
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 1, got %d", len(args))
	}

	tradeID := args[0]

	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	for _, trade := range ledger.Trades {
		if trade.ID == tradeID {
			return json.Marshal(trade)
		}
	}

	return nil, fmt.Errorf("Trade not found: %s", tradeID)
}

// getAllTrades retrieves all trades from the supply chain
func (s *SupplyChainFinance) getAllTrades(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Retrieving all trades")
	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	return json.Marshal(ledger.Trades)
}

// getTradeHistory retrieves the history of a trade from the supply chain
func (s *SupplyChainFinance) getTradeHistory(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Retrieving trade history")
	if len(args) != 1 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 1, got %d", len(args))
	}

	tradeID := args[0]

	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	var history []Trade
	for _, trade := range ledger.Trades {
		if trade.ID == tradeID {
			history = append(history, trade)
		}
	}

	return json.Marshal(history)
}

// verifyTrade verifies the integrity of a trade in the supply chain
func (s *SupplyChainFinance) verifyTrade(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	log.Println("Verifying trade")
	if len(args) != 1 {
		return nil, fmt.Errorf("Invalid number of arguments. Expecting 1, got %d", len(args))
	}

	tradeID := args[0]

	ledger, err := s.getLedger(stub)
	if err != nil {
		return nil, err
	}

	for _, trade := range ledger.Trades {
		if trade.ID == tradeID {
			if trade.Hash != calculateHash(trade) {
				return nil, fmt.Errorf("Trade has been tampered with: %s", tradeID)
			}
			return []byte("Trade is valid")
		}
	}

	return nil, fmt.Errorf("Trade not found: %s", tradeID)
}

// getLedger retrieves the ledger from the supply chain
func (s *SupplyChainFinance) getLedger(stub shim.ChaincodeStubInterface) (*Ledger, error) {
	log.Println("Retrieving ledger")
	ledgerBytes, err := stub.GetState("ledger")
	if err != nil {
		return nil, err
	}

	var ledger Ledger
	err = json.Unmarshal(ledgerBytes, &ledger)
	if err != nil {
		return nil, err
	}

	return &ledger, nil
}

// calculateHash calculates the hash of a trade
func calculateHash(trade Trade) string {
	log.Println("Calculating hash")
	hash := sha256.New()
	hash.Write([]byte(trade.ID))
	hash.Write([]byte(trade.Buyer))
	hash.Write([]byte(trade.Seller))
	hash.Write([]byte(trade.Product))
	hash.Write([]byte(strconv.Itoa(trade.Quantity)))
	hash.Write([]byte(trade.Price.String()))
	hash.Write([]byte(trade.Status))
	hash.Write([]byte(strconv.FormatInt(trade.Timestamp, 10)))

	return fmt.Sprintf("%x", hash.Sum(nil))
}

func main() {
	log.Println("Starting supply chain finance smart contract")
	contract := new(SupplyChainFinance)
	contract.TransactionContextHandler = new(contractapi.TransactionContext)
	contract.BeforeTransaction = func(ctx contractapi.TransactionContextInterface) error {
		log.Println("Before transaction")
		return nil
	}
	contract.AfterTransaction = func(ctx contractapi.TransactionContextInterface) error {
		log.Println("After transaction")
		return nil
	}

	cc, err := contractapi.NewChaincode(contract)
	if err != nil {
		log.Panicf("Error creating chaincode: %s", err)
	}

	if err := shim.Start(cc); err != nil {
		log.Panicf("Error starting chaincode: %s", err)
	}
}
