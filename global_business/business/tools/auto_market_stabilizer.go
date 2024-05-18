package main

import (
	"time"
)

// PiCoin struct to hold Pi Coin information
type PiCoin struct {
	Price      float64
	Launched   bool
	LaunchDate time.Time
}

func main() {
	// Create new Pi Coin
	piCoin := &PiCoin{
		Price:      314.159,
		Launched:   false,
		LaunchDate: time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC),
	}

	// Set Pi Coin price
	piCoin.setPrice()

	// Launch Pi Coin as stable coin
	piCoin.launch()
}

// setPrice sets the price of the Pi Coin
func (p *PiCoin) setPrice() {
	p.Price = 314.159
	println("Pi Coin price set to: $", p.Price)
}

// launch launches the Pi Coin as a stable coin
func (p *PiCoin) launch() {
	if time.Now().After(p.LaunchDate) {
		p.Launched = true
		println("Pi Coin launched as stable coin")
	} else {
		println("Pi Coin launch date not yet reached")
	}
}
