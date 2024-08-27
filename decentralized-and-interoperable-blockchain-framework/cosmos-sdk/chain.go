package chain

import (
	"fmt"

	"github.com/cosmos/cosmos-sdk/types"
)

// PiNetworkChain is a custom chain implementation for Pi Network
type PiNetworkChain struct {
	*types.Chain
}

func NewPiNetworkChain() *PiNetworkChain {
	return &PiNetworkChain{
		Chain: types.NewChain(
			"pi-network",
			"Pi Network",
			"PN",
			"pico",
			6,
			[]string{"https://rpc.pi-network.io"},
		),
	}
}

func (c *PiNetworkChain) GetGenesis() *types.Genesis {
	return &types.Genesis{
		GenesisTime:     time.Now(),
		ChainID:        c.Chain.ID,
		ConsensusParams: c.GetConsensusParams(),
		Validators:     c.GetValidators(),
		AppState:       c.GetAppState(),
	}
}

func (c *PiNetworkChain) GetConsensusParams() *types.ConsensusParams {
	return &types.ConsensusParams{
		BlockTime: 10 * time.Second,
		BlockSize: 1000000,
	}
}

func (c *PiNetworkChain) GetValidators() []*types.Validator {
	return []*types.Validator{
		{
			Address: "validator1",
			PubKey:  "pubkey1",
			Power:   10,
		},
		{
			Address: "validator2",
			PubKey:  "pubkey2",
			Power:   20,
		},
	}
}

func (c *PiNetworkChain) GetAppState() *types.AppState {
	return &types.AppState{
		Params: c.GetParams(),
	}
}

func (c *PiNetworkChain) GetParams() *types.Params {
	return &types.Params{
		MaxGas: 1000000,
	}
}
