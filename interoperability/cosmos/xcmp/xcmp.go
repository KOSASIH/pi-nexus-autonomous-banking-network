package xcmp

import (
	"fmt"

	"github.com/polkadot-network/polkadot/xcm"
)

type XCMP struct {
	// XCMP client instance
	client *xcm.Client
}

func NewXCMP() *XCMP {
	return &XCMP{
		client: xcm.NewClient(),
	}
}

func (x *XCMP) SendMessage(from, to string, message []byte) error {
	// Create a new XCMP message
	msg := xcm.NewMessage(from, to, message)

	// Send the message using the XCMP client
	err := x.client.SendMessage(ctx, msg)
	if err != nil {
		return fmt.Errorf("failed to send message: %s", err)
	}

	return nil
}

func (x *XCMP) ReceiveMessage() ([]byte, error) {
	// Receive a message using the XCMP client
	msg, err := x.client.ReceiveMessage(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to receive message: %s", err)
	}

	return msg, nil
}
