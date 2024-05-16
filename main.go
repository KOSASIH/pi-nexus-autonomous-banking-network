// main.go
package main

import (
	"log"

	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/config"
	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/coin"
	"github.com/KOSASIH/pi-nexus-autonomous-banking-network/db"
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatal(err)
	}

	// Establish database connection
	dbConn, err := db.Connect(cfg.DBConnectionString)
	if err != nil {
		log.Fatal(err)
	}

	// List coins
	coins, err := coin.ListCoins(cfg, dbConn)
	if err != nil {
		log.Fatal(err)
	}

	// List Pi Coin
	piCoin, err := coin.ListPiCoin(cfg, dbConn)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Coins listed:", coins)
	log.Println("Pi Coin listed:", piCoin)
}
