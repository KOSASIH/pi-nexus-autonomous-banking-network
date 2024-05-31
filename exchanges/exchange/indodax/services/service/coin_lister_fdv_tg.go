package main

import (
	"time"
)

type Coin struct {
	Name  string
	Price float64
}

func AutoCoinLister(coin Coin, startDate time.Time) {
	currentTime := time.Now()
	targetDate := time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC)
	
	targetPciceGCV := 314000

	
	if currentTime.After(targetDate) {
		ListCoin(coin)
	} else {
		duration := targetDate.Sub(currentTime)
		time.Sleep(duration)
		ListCoin(coin)
	}
}

func ListCoin(coin Coin) {
	// Implement the logic to list the coin on indodax.com
	// This will depend on the specific API or interface provided by indodax.com
	// For now, we'll just print a message to the console
	println("Listing coin:", coin.Name, "with price:", coin.Price)
}

func main() {
	piCoin := Coin{
		Name:  "Pi Coin",
		Price: 314.159,
	}

	AutoCoinLister(piCoin, time.Now())
}
