package main

import (
	"github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/x/bridge"
)

func main() {
	// Create a new bridge instance
	bridgeInstance, err := bridge.NewBridge(
		"pi-network",
		"cosmos-hub",
		"bridge-1",
		"bridge-2",
	)

	//...
}
