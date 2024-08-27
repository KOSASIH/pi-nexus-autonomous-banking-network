package bridge

import (
	"fmt"

	"github.com/polkadot-network/polkadot-sdk/types"
)

// PiNetworkBridge is a custom bridge implementation for Pi Network
type PiNetworkBridge struct {
	*types.Bridge
}

func NewPiNetworkBridge() *PiNetworkBridge {
	return &PiNetworkBridge{
		Bridge: types.NewBridge(
			"pi-network",
			"Pi Network",
			"PN",
			"pico",
			6,
			[]string{"https://rpc.pi-network.io"},
		),
	}
}

func (b *PiNetworkBridge) GetBridgeConfig() *types.BridgeConfig {
	return &types.BridgeConfig{
		ChainID:        b.Bridge.ChainID,
		BridgeID:       b.Bridge.BridgeID,
		Validators:     b.GetValidators(),
		Relayers:       b.GetRelayers(),
		ChainConfigs:   b.GetChainConfigs(),
		BridgeConfigs:  b.GetBridgeConfigs(),
	}
}

func (b *PiNetworkBridge) GetValidators() []*types.Validator {
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

func (b *PiNetworkBridge) GetRelayers() []*types.Relay {
	return []*types.Relay{
		{
			Address: "relayer1",
			PubKey:  "pubkey1",
			Power:   10,
		},
		{
			Address: "relayer2",
			PubKey:  "pubkey2",
			Power:   20,
		},
	}
}

func (b *PiNetworkBridge) GetChainConfigs() []*types.ChainConfig {
	return []*types.ChainConfig{
		{
			ChainID: "pi-network",
			RPC:     "https://rpc.pi-network.io",
		},
	}
}

func (b *PiNetworkBridge) GetBridgeConfigs() []*types.BridgeConfig {
	return []*types.BridgeConfig{
		{
			BridgeID: "pi-network-bridge",
			ChainID:  "pi-network",
			RPC:      "https://rpc.pi-network.io",
		},
	}
}
