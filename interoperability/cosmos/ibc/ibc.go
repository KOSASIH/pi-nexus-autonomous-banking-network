package ibc

import (
	"fmt"

	"github.com/cosmos/cosmos-sdk/codec"
	"github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/x/ibc"
	"github.com/cosmos/cosmos-sdk/x/ibc/04-channel"
	"github.com/cosmos/cosmos-sdk/x/ibc/23-commitment"
)

type IBC struct {
	// codec to marshal/unmarshal data
	cdc *codec.Codec
	// IBC module instance
	ibcModule *ibc.Module
}

func NewIBC(cdc *codec.Codec) *IBC {
	return &IBC{
		cdc:     cdc,
		ibcModule: ibc.NewModule(),
	}
}

func (i *IBC) RegisterCodec() {
	i.ibcModule.RegisterCodec(i.cdc)
	commitment.RegisterCodec(i.cdc)
	channel.RegisterCodec(i.cdc)
}

func (i *IBC) CreateChannel(portID, channelID string) error {
	// Create a new channel capability for sending and receiving packets
	channelCap := channel.NewCapability(&i.ibcModule)

	// Create a new channel endpoint
	localChannel := channel.NewChannel(
		portID,
		channelID,
		channel.ORDERED,
		channelCap,
		true,
	)

	// Create a connection between the local chain and the remote chain
	connectionID := ibc.NewConnectionID("exampleconnection")
	connection := ibc.NewConnectionEnd(
		ibc.INIT,
		ibc.ExportedVersionsToProto(ibc.SupportedVersions),
		true,
	)

	// Initialize the connection and add it to the IBC module
	err := i.ibcModule.ConnInit(ctx, connectionID, connection)
	if err != nil {
		return fmt.Errorf("failed to initialize connection: %s", err)
	}

	// Open the channel between the local and remote chains
	err = i.ibcModule.ChanOpenInit(ctx, connectionID, channelCap, localChannel)
	if err != nil {
		return fmt.Errorf("failed to open channel: %s", err)
	}

	return nil
}
