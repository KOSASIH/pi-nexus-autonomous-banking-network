// pi_network_api.go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
)

type Transaction struct {
	Sender    string  `json:"sender"`
	Recipient string  `json:"recipient"`
	Amount    float64 `json:"amount"`
}

type Block struct {
	Index      int         `json:"index"`
	Timestamp  string      `json:"timestamp"`
	Transactions []Transaction `json:"transactions"`
	Hash        string      `json:"hash"`
	PrevHash    string      `json:"prev_hash"`
}

func main() {
	r := mux.NewRouter()

	r.HandleFunc("/transactions", createTransaction).Methods("POST")
	r.HandleFunc("/blocks", getBlocks).Methods("GET")
	r.HandleFunc("/blocks/{blockId}", getBlock).Methods("GET")
	r.HandleFunc("/blocks", createBlock).Methods("POST")

	http.ListenAndServe(":8080", r)
}

func createTransaction(w http.ResponseWriter, r *http.Request) {
	var tx Transaction
	err := json.NewDecoder(r.Body).Decode(&tx)
	if err!= nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Create new transaction and add to mempool
	fmt.Fprint(w, "Transaction created successfully")
}

func getBlocks(w http.ResponseWriter, r *http.Request) {
	// Return list of blocks
	json.NewEncoder(w).Encode([]Block{{Index: 1, Timestamp: "2023-02-20T14:30:00", Transactions: [...], Hash: "0x...", PrevHash: "0x..."}})
}

func getBlock(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	blockId, err := strconv.Atoi(vars["blockId"])
	if err!= nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Return block by ID
	json.NewEncoder(w).Encode(Block{Index: blockId, Timestamp: "2023-02-20T14:30:00", Transactions: [...], Hash: "0x...", PrevHash: "0x..."})
}

func createBlock(w http.ResponseWriter, r *http.Request) {
	// Create new block and add to blockchain
	fmt.Fprint(w, "Block created successfully")
}
