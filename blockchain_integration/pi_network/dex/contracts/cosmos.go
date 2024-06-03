package main

import (
	"fmt"

	"github.com/cosmos/cosmos-sdk/types"
)

type PolkadotModule struct{}

func (pm *PolkadotModule) InitGenesis(ctx sdk.Context, data json.RawMessage) []abci.ValidatorUpdate {
	return nil
}

func (pm *PolkadotModule) BeginBlock(ctx sdk.Context, req abci.RequestBeginBlock) {
}

func (pm *PolkadotModule) EndBlock(ctx sdk.Context, req abci.RequestEndBlock) []abci.ValidatorUpdate {
	return nil
}

func (pm *PolkadotModule) Deposit(ctx sdk.Context, amount uint64) {
	fmt.Println("Deposit:", amount)
}

func (pm *PolkadotModule) Withdraw(ctx sdk.Context, amount uint64) {
	fmt.Println("Withdrawal:", amount)
}
