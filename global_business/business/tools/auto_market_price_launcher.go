package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"time"

	"github.com/dgryski/go-farm"
	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

// PiCoin struct represents the cryptocurrency
type PiCoin struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Symbol    string `json:"symbol"`
	TotalSupply  *big.Int `json:"total_supply"`
	MarketCap  *big.Int `json:"market_cap"`
	PriceUSD  *big.Float `json:"price_usd"`
	LaunchDate time.Time `json:"launch_date"`
}

// GlobalPiCoin instance
var GlobalPiCoin PiCoin

func init() {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}

	// Initialize PiCoin instance
	GlobalPiCoin = PiCoin{
		ID:        "pi-coin",
		Name:      "Pi Coin",
		Symbol:    "π",
		TotalSupply: big.NewInt(1000000000), // 1 billion coins
		MarketCap: big.NewInt(314159000000), // $314.159 billion
		PriceUSD: big.NewFloat(314.159),
		LaunchDate: time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC),
	}
}

func main() {
	// Create a new HTTP router
	r := mux.NewRouter()

	// Define API endpoints
	r.HandleFunc("/pi-coin/price", getPriceHandler).Methods("GET")
	r.HandleFunc("/pi-coin/market-cap", getMarketCapHandler).Methods("GET")
	r.HandleFunc("/pi-coin/launch", launchHandler).Methods("POST")

	// Start the HTTP server
	log.Fatal(http.ListenAndServe(":8080", r))
}

// getPriceHandler returns the current price of Pi Coin in USD
func getPriceHandler(w http.ResponseWriter, r *http.Request) {
	priceJSON, err := json.Marshal(GlobalPiCoin.PriceUSD)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(priceJSON)
}

// getMarketCapHandler returns the current market capitalization of Pi Coin
func getMarketCapHandler(w http.ResponseWriter, r *http.Request) {
	marketCapJSON, err := json.Marshal(GlobalPiCoin.MarketCap)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(marketCapJSON)
}

// launchHandler sets the global market price and launches Pi Coin
func launchHandler(w http.ResponseWriter, r *http.Request) {
	// Generate a cryptographic key pair for Pi Coin
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	publicKey := privateKey.Public()

	// Create a digital signature for the launch event
	signature, err := signLaunchEvent(privateKey, publicKey)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the global market price and launch Pi Coin
	GlobalPiCoin.PriceUSD = big.NewFloat(314.159)
	GlobalPiCoin.LaunchDate = time.Now()

	// Broadcast the launch event to the world
	broadcastLaunchEvent(signature)

	w.WriteHeader(http.StatusCreated)
	fmt.Fprint(w, "Pi Coin launched successfully!")
}

// signLaunchEvent generates a digital signature for the launch event
func signLaunchEvent(privateKey *ecdsa.PrivateKey, publicKey *ecdsa.PublicKey) ([]byte, error) {
	launchEvent := fmt.Sprintf("Pi Coin launch event on %s", GlobalPiCoin.LaunchDate.Format(time.RFC3339))
	hash := sha256.Sum256([]byte(launchEvent))
	signature, err := privateKey.Sign(rand.Reader, hash[:], nil)
	return signature, err
}

// broadcastLaunchEvent broadcasts the launch event to the world
func broadcastLaunchEvent(signature []byte) {
	// Simulate broadcasting the launch event to various channels (e.g., social media, news outlets, etc.)
	fmt.Println("Broadcasting Pi Coin launch event...")
	fmt.Printf("Launch event: %s\n", GlobalPiCoin.LaunchDate.Format(time.RFC3339))
	fmt.Printf("Signature: %x\n", signature)
}
